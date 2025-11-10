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

from app.ml.adv.decision_net import decision_net_score
from app.ml.adv.options_vol_model import options_vol_edge
from app.ml.adv.macro_onchain_model import macro_onchain_bias

from app.db.models import FundingRate, OptionsDerivedMetrics, MacroData, OnchainMetrics

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
       
        if not self.is_ready:
            raise RuntimeError("InferenceService not ready. Check model loading logs.")

        db = SessionLocal()
        try:
            # -----------------------------
            # 1) Load recent feature window
            # -----------------------------
            total_lookback = SEQ_LEN + max(ROLL_WINDOWS) + 5
            raw_df = self._fetch_inference_data(db, ctx.symbol, total_lookback)

            if raw_df is None or len(raw_df) < total_lookback:
                return HybridDecision(
                    symbol=ctx.symbol,
                    instrument_type=ctx.instrument_type,
                    direction="flat",
                    p_edge=0.0,
                    confidence=0.0,
                    size_factor=0.0,
                    strategy_tag="no_data",
                    meta_execute=False,
                    debug={"rows": 0 if raw_df is None else len(raw_df)},
                )

            # -----------------------------
            # 2) Prepare inputs
            # -----------------------------
            x_blocks, x_tabular = self._prepare_inputs(raw_df)

            # -----------------------------
            # 3) Core experts: TFT, TCN, XGB
            # -----------------------------
            with torch.no_grad():
                tft_out = self.models["tft"](x_blocks)
                tcn_out = self.models["tcn"](x_blocks)

                xgb_price = (
                    _xgb_predict(self.models["xgb_price"], x_tabular)
                    if self.models.get("xgb_price") is not None
                    else None
                )
                xgb_vol = (
                    _xgb_predict(self.models["xgb_vol"], x_tabular)
                    if self.models.get("xgb_vol") is not None
                    else None
                )

            expert = ExpertSignals(
                tft_price=float(tft_out["price"].item()),
                tcn_price=float(tcn_out["price"].item()),
                xgb_price=xgb_price,
                tft_vol=float(tft_out["vol"].item()) if "vol" in tft_out else None,
                tcn_vol=float(tcn_out["vol"].item()) if "vol" in tcn_out else None,
                xgb_vol=xgb_vol,
            )

            # --------------------------------------
            # 4) Base context features (vol & trend)
            # --------------------------------------
            ret = raw_df["close"].pct_change()

            rv_24h = float(ret.rolling(96).std().iloc[-1] or 0.0)        # ~24h window
            trend_score = float(ret.rolling(48).mean().iloc[-1] or 0.0)  # short/mid bias

            # Funding proxy from DB (if available)
            try:
                funding_rows = (
                    db.query(FundingRate)
                    .filter(FundingRate.symbol == ctx.symbol)
                    .order_by(FundingRate.timestamp.desc())
                    .limit(4)
                    .all()
                )
                if funding_rows:
                    funding_1h = float(
                        sum(fr.funding_rate for fr in funding_rows) / len(funding_rows)
                    )
                else:
                    funding_1h = 0.0
            except Exception:
                funding_1h = 0.0

            features: Dict[str, Any] = {
                "rv_24h": rv_24h,
                "funding_1h": funding_1h,
                "trend_score": trend_score,
            }

            # -----------------------------
            # 5) Specialists
            # -----------------------------

            def _underlying_from_symbol(sym: str) -> str:
                base = sym.split("-")[0]
                if "/" in base:
                    base = base.split("/")[0]
                return base

            underlying = _underlying_from_symbol(ctx.symbol)

            # 5a) DecisionNet specialist
            decision_features = {
                "tft_price": expert.tft_price or 0.0,
                "tcn_price": expert.tcn_price or 0.0,
                "xgb_price": expert.xgb_price or 0.0,
                "tft_vol": expert.tft_vol or 0.0,
                "tcn_vol": expert.tcn_vol or 0.0,
                "xgb_vol": expert.xgb_vol or 0.0,
                "rv_24h": features["rv_24h"],
                "trend_score": features["trend_score"],
                "funding_1h": features["funding_1h"],
            }
            dec_score = decision_net_score(decision_features)

            # 5b) Options Vol/Skew specialist
            options_features: Dict[str, Any] = {}
            try:
                odm = (
                    db.query(OptionsDerivedMetrics)
                    .filter(OptionsDerivedMetrics.symbol == underlying)
                    .order_by(OptionsDerivedMetrics.timestamp.desc())
                    .first()
                )
                if odm:
                    iv_mid = odm.avg_iv_mid_term or odm.avg_iv_near_term or 0.0
                    iv_rank = max(0.0, min(1.0, iv_mid / 200.0))  # crude normalization
                    rr_25d = odm.iv_skew_25d or 0.0
                    term_slope = (
                        getattr(odm, "iv_term_slope_near_mid", None)
                        or getattr(odm, "iv_term_slope_reg_logT", None)
                        or 0.0
                    )

                    options_features = {
                        "iv_rank": float(iv_rank),
                        "risk_reversal_25d": float(rr_25d),
                        "term_structure_slope": float(term_slope),
                    }
            except Exception:
                options_features = {}

            opt_edge = options_vol_edge(options_features) if options_features else 0.0

            # 5c) Macro + On-chain specialist
            macro_features: Dict[str, Any] = {}
            try:
                # MacroData: indicators like 'DXY', 'SPX'
                def _macro_trend(indicator: str, limit: int = 10) -> float:
                    rows = (
                        db.query(MacroData)
                        .filter(MacroData.indicator == indicator)
                        .order_by(MacroData.timestamp.desc())
                        .limit(limit)
                        .all()
                    )
                    if len(rows) < 2:
                        return 0.0
                    latest = rows[0].value
                    oldest = rows[-1].value
                    if not oldest:
                        return 0.0
                    return float((latest - oldest) / abs(oldest))

                dxy_trend = _macro_trend("DXY")
                spx_trend = _macro_trend("SPX")

                oc = (
                    db.query(OnchainMetrics)
                    .filter(OnchainMetrics.symbol == underlying)
                    .order_by(OnchainMetrics.timestamp.desc())
                    .first()
                )

                stablecoin_netflow = float(
                    getattr(oc, "exchange_net_flow_usd", 0.0)
                ) if oc else 0.0

                btc_exchange_reserves_change = 0.0  # not modeled yet, keep neutral

                macro_features = {
                    "stablecoin_netflow": stablecoin_netflow,
                    "btc_exchange_reserves_change": btc_exchange_reserves_change,
                    "dxy_trend": dxy_trend,
                    "spx_trend": spx_trend,
                }
            except Exception:
                macro_features = {}

            macro_bias = macro_onchain_bias(macro_features) if macro_features else 0.0

            # Attach specialist outputs
            expert.decision_net_score = float(dec_score)
            expert.options_vol_edge = float(opt_edge)
            expert.macro_onchain_bias = float(macro_bias)

            # -----------------------------
            # 6) Meta-ensemble
            # -----------------------------
            meta = meta_predict(features, expert, ctx)

            # -----------------------------
            # 7) Meta-label (execute? size?)
            # -----------------------------
            ml = metalabel_decide(features, expert, meta, ctx)

            if not ml.get("execute", False):
                # Meta-label veto → stay flat, expose diagnostics
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
                        "specialists": {
                            "decision_net_score": dec_score,
                            "options_vol_edge": opt_edge,
                            "macro_onchain_bias": macro_bias,
                            "macro_features": macro_features,
                            "options_features": options_features,
                        },
                    },
                )

            # -----------------------------
            # 8) Bandit / regime router
            # -----------------------------
            weights, tag = bandit_weights(features, expert, meta, ctx)

            # -----------------------------
            # 9) Final decision payload
            # -----------------------------
            return HybridDecision(
                symbol=ctx.symbol,
                instrument_type=ctx.instrument_type,
                direction=meta["dir_raw"],
                p_edge=meta["p_edge"],
                confidence=meta["confidence"],
                size_factor=float(ml.get("size_factor", 0.0)),
                strategy_tag=tag,
                meta_execute=True,
                debug={
                    "expert": expert.dict(),
                    "meta": meta,
                    "specialists": {
                        "decision_net_score": dec_score,
                        "options_vol_edge": opt_edge,
                        "macro_onchain_bias": macro_bias,
                        "macro_features": macro_features,
                        "options_features": options_features,
                    },
                    "weights": weights,
                },
            )

        except Exception as e:
            logger.error(f"Hybrid decision failed for {ctx.symbol}: {e}", exc_info=True)
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
