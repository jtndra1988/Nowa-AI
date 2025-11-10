import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from app.db.database import SessionLocal
from app.core.config import settings

from app.ml.adv.models_tft import TemporalFusionTransformer
from app.ml.adv.models_tcn import TemporalConvNet
from app.ml.adv.ensemble import StackingEnsemble
from app.ml.adv.feature_engineering import create_tabular_features

from app.hybrid.schemas import MarketContext, ExpertSignals, HybridDecision
from app.hybrid.meta_ensemble import meta_predict
from app.hybrid.metalabel import metalabel_decide
from app.hybrid.bandit import bandit_weights

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("./model_artifacts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 60

FEATURE_BLOCKS = {
    "price": ["open", "high", "low", "close", "volume"],
    "ob": ["ob_imbalance_1s", "ob_spread", "ob_depth_ask_1", "ob_depth_bid_1"],
    "sent": ["sent_score_1m", "sent_score_15m"],
}

TABULAR_FEATURE_COLS = ["close", "volume", "open", "high", "low"]
ROLL_WINDOWS = [5, 10, 20]

TFT_CONFIG = {
    "seq_len": SEQ_LEN,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dropout": 0.1,
}
TFT_WEIGHTS_PATH = ARTIFACT_DIR / "tft_best_model.pth"
SCALER_PATH = ARTIFACT_DIR / "tft_scaler.npz"

TCN_CONFIG = {
    "channels": (64, 128, 128),
    "kernel": 3,
    "dropout": 0.1,
}
TCN_WEIGHTS_PATH = ARTIFACT_DIR / "tcn_best_model.pth"

XGB_PRICE_PATH = ARTIFACT_DIR / "xgb_price_model.joblib"
XGB_VOL_PATH = ARTIFACT_DIR / "xgb_vol_model.joblib"

ENSEMBLE_CHECKPOINT_PATH = ARTIFACT_DIR / "hybrid_ensemble_best.pth"
N_MODELS = 3  # TFT, TCN, XGB


def _xgb_predict(xgb_model, x_tabular_df: pd.DataFrame) -> float:
    pred = xgb_model.predict(x_tabular_df)
    return float(pred[0])


class HybridInferenceService:
    def __init__(self):
        logger.info(f"Initializing HybridInferenceService on {DEVICE}")
        self.models: Dict[str, Any] = {}
        self.is_ready = False

        try:
            # Scaler
            if SCALER_PATH.exists():
                scaler = np.load(SCALER_PATH)
                self.scaler_mean = scaler["mean"]
                self.scaler_std = scaler["std"]
            else:
                logger.warning("Scaler not found; using identity scaling.")
                self.scaler_mean = 0.0
                self.scaler_std = 1.0

            # TFT
            feature_dims = {b: len(c) for b, c in FEATURE_BLOCKS.items()}
            tft = TemporalFusionTransformer(feature_dims=feature_dims, **TFT_CONFIG).to(DEVICE)
            if TFT_WEIGHTS_PATH.exists():
                tft.load_state_dict(torch.load(TFT_WEIGHTS_PATH, map_location=DEVICE))
            else:
                logger.warning("TFT weights not found; using random init.")
            tft.eval()
            self.models["tft"] = tft

            # TCN
            total_in_feat = sum(len(c) for c in FEATURE_BLOCKS.values())
            tcn = TemporalConvNet(in_feat=total_in_feat, **TCN_CONFIG).to(DEVICE)
            if TCN_WEIGHTS_PATH.exists():
                tcn.load_state_dict(torch.load(TCN_WEIGHTS_PATH, map_location=DEVICE))
            else:
                logger.warning("TCN weights not found; using random init.")
            tcn.eval()
            self.models["tcn"] = tcn

            # XGB
            if XGB_PRICE_PATH.exists():
                self.models["xgb_price"] = joblib.load(XGB_PRICE_PATH)
            else:
                logger.warning("XGB price model missing.")
                self.models["xgb_price"] = None

            if XGB_VOL_PATH.exists():
                self.models["xgb_vol"] = joblib.load(XGB_VOL_PATH)
            else:
                logger.warning("XGB vol model missing.")
                self.models["xgb_vol"] = None

            # Blenders
            if ENSEMBLE_CHECKPOINT_PATH.exists():
                bw = torch.load(ENSEMBLE_CHECKPOINT_PATH, map_location=DEVICE)

                pb = StackingEnsemble(n_models=N_MODELS).to(DEVICE)
                pb.load_state_dict(bw["price_blender_state_dict"])
                pb.eval()
                self.models["blender_price"] = pb

                vb = StackingEnsemble(n_models=N_MODELS).to(DEVICE)
                vb.load_state_dict(bw["vol_blender_state_dict"])
                vb.eval()
                self.models["blender_vol"] = vb
            else:
                logger.warning("Ensemble checkpoint missing; using simple averages.")
                self.models["blender_price"] = None
                self.models["blender_vol"] = None

            self.is_ready = True
            logger.info("HybridInferenceService initialized.")
        except Exception as e:
            self.is_ready = False
            logger.error(f"Failed to init HybridInferenceService: {e}", exc_info=True)
            raise

    # ---------- Data & features ----------

    def _fetch_inference_data(self, db, symbol: str, lookback: int) -> pd.DataFrame:
        logger.info(f"Fetching inference data for {symbol}")
        query = """
        SELECT
          m.timestamp, m.symbol,
          m.open, m.high, m.low, m.close, m.volume,
          ob.ob_imbalance_1s, ob.ob_spread, ob.ob_depth_ask_1, ob.ob_depth_bid_1,
          sf.sent_score_1m, sf.sent_score_15m
        FROM futures_market_data m
        LEFT JOIN orderbook_data ob
          ON m.timestamp = ob.timestamp AND m.symbol = ob.symbol
        LEFT JOIN sentiment_fusion sf
          ON m.timestamp = sf.timestamp AND m.symbol = sf.symbol
        WHERE m.symbol = :symbol
        ORDER BY m.timestamp DESC
        LIMIT :lookback
        """
        df = pd.read_sql(
            query,
            db.bind,
            params={"symbol": symbol, "lookback": lookback},
            parse_dates=["timestamp"],
        )
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        return df

    def _prepare_inputs(self, df: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
        # Tabular
        tabular_df = create_tabular_features(df, TABULAR_FEATURE_COLS, ROLL_WINDOWS)
        if tabular_df.empty:
            raise ValueError("Tabular features empty.")
        x_tabular = tabular_df.iloc[[-1]]

        # Sequential
        x_seq_df = pd.DataFrame()
        for _, cols in FEATURE_BLOCKS.items():
            x_seq_df = pd.concat([x_seq_df, df[cols]], axis=1)

        x_seq_raw = x_seq_df.iloc[-SEQ_LEN:].values.astype(np.float32)
        x_seq_scaled = (x_seq_raw - self.scaler_mean) / (self.scaler_std + 1e-8)

        x_blocks: Dict[str, torch.Tensor] = {}
        start = 0
        for name, cols in FEATURE_BLOCKS.items():
            end = start + len(cols)
            block = x_seq_scaled[:, start:end]
            x_blocks[name] = (
                torch.from_numpy(block)
                .float()
                .unsqueeze(0)
                .to(DEVICE)
            )
            start = end

        return x_blocks, x_tabular

    # ---------- Public: forecasts (existing behavior) ----------

    def predict(self, symbol: str) -> Dict[str, float]:
        if not self.is_ready:
            raise RuntimeError("InferenceService not ready.")
        db = SessionLocal()
        try:
            lookback = SEQ_LEN + max(ROLL_WINDOWS) + 5
            df = self._fetch_inference_data(db, symbol, lookback)
            if len(df) < lookback:
                raise ValueError(f"Not enough data for {symbol}")

            x_blocks, x_tabular = self._prepare_inputs(df)

            with torch.no_grad():
                tft_out = self.models["tft"](x_blocks)          # dict
                tcn_out = self.models["tcn"](x_blocks)          # dict

                xgb_price = (
                    _xgb_predict(self.models["xgb_price"], x_tabular)
                    if self.models["xgb_price"] is not None
                    else float(tft_out["price"][0].item())
                )
                xgb_vol = (
                    _xgb_predict(self.models["xgb_vol"], x_tabular)
                    if self.models["xgb_vol"] is not None
                    else float(tft_out["vol"][0].item())
                )

                price_vec = torch.tensor(
                    [
                        tft_out["price"][0].item(),
                        tcn_out["price"][0].item(),
                        xgb_price,
                    ],
                    device=DEVICE,
                ).unsqueeze(0)

                vol_vec = torch.tensor(
                    [
                        tft_out["vol"][0].item(),
                        tcn_out["vol"][0].item(),
                        xgb_vol,
                    ],
                    device=DEVICE,
                ).unsqueeze(0)

                if self.models["blender_price"] is not None:
                    final_price = self.models["blender_price"](price_vec).item()
                else:
                    final_price = float(price_vec.mean().item())

                if self.models["blender_vol"] is not None:
                    final_vol = self.models["blender_vol"](vol_vec).item()
                else:
                    final_vol = float(vol_vec.mean().item())

                # TFT feature weights: [B,L,F] -> avg
                fw = (
                    tft_out["feature_weights"]
                    .mean(dim=(0, 1))
                    .detach()
                    .cpu()
                    .numpy()
                )
                feature_importance = {
                    name: float(w)
                    for name, w in zip(FEATURE_BLOCKS.keys(), fw)
                }

                return {
                    "price_prediction": final_price,
                    "volatility_prediction": final_vol,
                    "feature_importance": feature_importance,
                    "tft_price": float(tft_out["price"][0].item()),
                    "tcn_price": float(tcn_out["price"][0].item()),
                    "xgb_price": float(xgb_price),
                }
        finally:
            db.close()

    # ---------- Public: hybrid decision for trading ----------

    def build_decision(self, ctx: MarketContext) -> HybridDecision:
        """
        Build a trade-ready hybrid decision from TFT + TCN + XGB.
        This is what Nowa should consume.

        Works for spot / perps / futures / options:
        - We keep instrument_type metadata here.
        - Position sizing / leverage logic can be applied downstream per type.
        """
        if not self.is_ready:
            raise RuntimeError("InferenceService not ready. Check model loading logs.")

        db = SessionLocal()
        try:
            # Similar lookback as predict()
            total_lookback = SEQ_LEN + max(ROLL_WINDOWS) + 5
            raw_df = self._fetch_inference_data(db, ctx.symbol, total_lookback)

            if len(raw_df) < total_lookback:
                return HybridDecision(
                    symbol=ctx.symbol,
                    instrument_type=ctx.instrument_type,
                    direction="flat",
                    p_edge=0.0,
                    confidence=0.0,
                    size_factor=0.0,
                    strategy_tag="no_data",
                    meta_execute=False,
                    debug={"rows": len(raw_df)},
                )

            # Reuse same input pipeline
            x_blocks, x_tabular = self._prepare_inputs(raw_df)

            with torch.no_grad():
                preds_tft = self.models["tft"](x_blocks)
                preds_tcn = self.models["tcn"](x_blocks)

                pred_xgb_price = (
                    _xgb_predict(self.models["xgb_price"], x_tabular)
                    if self.models.get("xgb_price") is not None
                    else None
                )
                pred_xgb_vol = (
                    _xgb_predict(self.models["xgb_vol"], x_tabular)
                    if self.models.get("xgb_vol") is not None
                    else None
                )

            expert = ExpertSignals(
                tft_price=float(preds_tft["price"].item()),
                tcn_price=float(preds_tcn["price"].item()),
                xgb_price=pred_xgb_price,
                tft_vol=float(preds_tft["vol"].item()) if "vol" in preds_tft else None,
                tcn_vol=float(preds_tcn["vol"].item()) if "vol" in preds_tcn else None,
                xgb_vol=pred_xgb_vol,
            )

            # Build simple context features
            ret = raw_df["close"].pct_change()
            rv_24h = float(ret.rolling(96).std().iloc[-1] or 0.0)      # ~24h on 15m bars
            trend_score = float(ret.rolling(48).mean().iloc[-1] or 0.0)

            features = {
                "rv_24h": rv_24h,
                "funding_1h": 0.0,          # plug real funding when available
                "trend_score": trend_score,
            }

            # 1) meta-ensemble: how strong is the edge, and which way?
            meta = meta_predict(features, expert, ctx)

            # 2) meta-label: filter out garbage, size the trade
            ml = metalabel_decide(features, expert, meta, ctx)

            if not ml["execute"]:
                return HybridDecision(
                    symbol=ctx.symbol,
                    instrument_type=ctx.instrument_type,
                    direction="flat",
                    p_edge=meta["p_edge"],
                    confidence=meta["confidence"],
                    size_factor=0.0,
                    strategy_tag="filtered",
                    meta_execute=False,
                    debug={
                        "expert": expert.dict(),
                        "meta": meta,
                    },
                )

            # 3) contextual bandit: regime tag + diagnostic weights
            weights, tag = bandit_weights(features, expert, meta, ctx)

            # Final direction from meta-ensemble; sizing from meta-label
            return HybridDecision(
                symbol=ctx.symbol,
                instrument_type=ctx.instrument_type,
                direction=meta["dir_raw"],
                p_edge=meta["p_edge"],
                confidence=meta["confidence"],
                size_factor=ml["size_factor"],
                strategy_tag=tag,
                meta_execute=True,
                debug={
                    "expert": expert.dict(),
                    "meta": meta,
                    "weights": weights,
                },
            )

        except Exception as e:
            self.logger.error(
                f"Hybrid decision failed for {ctx.symbol}: {e}",
                exc_info=True,
            )
            return HybridDecision(
                symbol=ctx.symbol,
                instrument_type=ctx.instrument_type,
                direction="flat",
                p_edge=0.0,
                confidence=0.0,
                size_factor=0.0,
                strategy_tag="error",
                meta_execute=False,
                debug={"error": str(e)},
            )
        finally:
            db.close()


try:
    inference_service = HybridInferenceService()
except Exception as e:
    inference_service = None
    logger.critical(f"Failed to initialize InferenceService on startup: {e}")
