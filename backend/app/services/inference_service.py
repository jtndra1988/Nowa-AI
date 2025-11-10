import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any
import logging
import warnings

# --- Import all our new ML components ---

from app.db.database import SessionLocal # For querying data
from app.core.config import settings
from app.hybrid.schemas import MarketContext, ExpertSignals, HybridDecision
from app.hybrid.meta_ensemble import meta_predict
from app.hybrid.metalabel import metalabel_decide
from app.hybrid.bandit import bandit_weights
# --- Suppress warnings ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATIONS (Must match training scripts) ---
# TODO: Ensure these match your training configurations
ARTIFACT_DIR = Path("./model_artifacts")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 60 # The sequence length our models were trained on

# 1. Sequential Features (for TFT/TCN)
FEATURE_BLOCKS = {
    "price": ['open', 'high', 'low', 'close', 'volume'],
    "ob": ['ob_imbalance_1s', 'ob_spread', 'ob_depth_ask_1', 'ob_depth_bid_1'],
    "sent": ['sent_score_1m', 'sent_score_15m'],
}

# 2. Tabular Features (for XGBoost)
TABULAR_FEATURE_COLS = ['close', 'volume', 'open', 'high', 'low']
ROLL_WINDOWS = [5, 10, 20]

# 3. TFT Model Config
TFT_CONFIG = {
    "seq_len": SEQ_LEN,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dropout": 0.1
}
TFT_WEIGHTS_PATH = ARTIFACT_DIR / "tft_best_model.pth"
SCALER_PATH = ARTIFACT_DIR / "tft_scaler.npz"

# 4. TCN Model Config
TCN_CONFIG = {
    "channels": (64, 128, 128),
    "kernel": 3,
    "dropout": 0.1
}
TCN_WEIGHTS_PATH = ARTIFACT_DIR / "tcn_best_model.pth"

# 5. XGBoost Model Config
XGB_PRICE_PATH = ARTIFACT_DIR / "xgb_price_model.joblib"
XGB_VOL_PATH = ARTIFACT_DIR / "xgb_vol_model.joblib"

# 6. Ensemble Blender Config
ENSEMBLE_CHECKPOINT_PATH = ARTIFACT_DIR / "hybrid_ensemble_best.pth"
N_MODELS = 3 # TFT, TCN, XGBoost

# --- Helper function ---
def _xgb_predict(xgb_model, x_tabular_df: pd.DataFrame) -> float:
    """ Helper to run a single prediction on a numpy-based XGB model """
    pred = xgb_model.predict(x_tabular_df)
    return float(pred[0])


class HybridInferenceService:
    """
    This service loads all hybrid model components (TFT, TCN, XGB, Blenders)
    on startup and provides a single method to run inference.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing HybridInferenceService on device: {DEVICE}")
        
        # --- Load all models ---
        self.models: Dict[str, Any] = {}
        
        try:
            # --- 1. Load Scaler ---
            self.logger.info(f"Loading scaler from {SCALER_PATH}")
            scaler = np.load(SCALER_PATH)
            self.scaler_mean = scaler['mean']
            self.scaler_std = scaler['std']

            # --- 2. Load TFT ---
            self.logger.info(f"Loading TFT model from {TFT_WEIGHTS_PATH}")
            # Get feature dims from config
            feature_dims = {bname: len(cols) for bname, cols in FEATURE_BLOCKS.items()}
            tft_model = TemporalFusionTransformer(feature_dims=feature_dims, **TFT_CONFIG).to(DEVICE)
            tft_model.load_state_dict(torch.load(TFT_WEIGHTS_PATH, map_location=DEVICE))
            tft_model.eval()
            self.models['tft'] = tft_model
            
            # --- 3. Load TCN ---
            self.logger.info(f"Loading TCN model from {TCN_WEIGHTS_PATH}")
            # Get total feature count
            total_in_feat = sum(len(cols) for cols in FEATURE_BLOCKS.values())
            tcn_model = TemporalConvNet(in_feat=total_in_feat, **TCN_CONFIG).to(DEVICE)
            tcn_model.load_state_dict(torch.load(TCN_WEIGHTS_PATH, map_location=DEVICE))
            tcn_model.eval()
            self.models['tcn'] = tcn_model

            # --- 4. Load XGB Models ---
            self.logger.info(f"Loading XGB models from {XGB_PRICE_PATH} and {XGB_VOL_PATH}")
            self.models['xgb_price'] = joblib.load(XGB_PRICE_PATH)
            self.models['xgb_vol'] = joblib.load(XGB_VOL_PATH)

            # --- 5. Load Blenders ---
            self.logger.info(f"Loading ensemble blenders from {ENSEMBLE_CHECKPOINT_PATH}")
            blender_weights = torch.load(ENSEMBLE_CHECKPOINT_PATH, map_location=DEVICE)
            
            price_blender = StackingEnsemble(n_models=N_MODELS).to(DEVICE)
            price_blender.load_state_dict(blender_weights['price_blender_state_dict'])
            price_blender.eval()
            self.models['blender_price'] = price_blender
            
            vol_blender = StackingEnsemble(n_models=N_MODELS).to(DEVICE)
            vol_blender.load_state_dict(blender_weights['vol_blender_state_dict'])
            vol_blender.eval()
            self.models['blender_vol'] = vol_blender
            
            self.is_ready = True
            self.logger.info("HybridInferenceService initialized successfully.")

        except Exception as e:
            self.is_ready = False
            self.logger.error(f"Failed to initialize HybridInferenceService: {e}", exc_info=True)
            raise e

    def _fetch_inference_data(self, db, symbol: str, lookback: int) -> pd.DataFrame:
        """
        Fetches and joins all required features from the database.
        This re-implements the logic lost from the old `run` function.
        
        NOTE: This query is complex and assumes your table structure.
        """
        self.logger.info(f"Fetching inference data for {symbol}...")
        
        # This SQL query joins all feature tables. It is the most critical
        # and most fragile part of the inference pipeline.
        # We need enough data to generate lags/rolling windows + the sequence.
        # `lookback` should be at least (SEQ_LEN + max(ROLL_WINDOWS))
        query = f"""
        SELECT 
            m.timestamp, m.symbol, m.open, m.high, m.low, m.close, m.volume,
            ob.ob_imbalance_1s, ob.ob_spread, ob.ob_depth_ask_1, ob.ob_depth_bid_1,
            sf.sent_score_1m, sf.sent_score_15m
        FROM 
            futures_market_data m
        LEFT JOIN 
            orderbook_data ob ON m.timestamp = ob.timestamp AND m.symbol = ob.symbol
        LEFT JOIN 
            sentiment_fusion sf ON m.timestamp = sf.timestamp AND m.symbol = sf.symbol
        WHERE 
            m.symbol = :symbol
        ORDER BY 
            m.timestamp DESC
        LIMIT :lookback
        """
        
        df = pd.read_sql(
            query, 
            db.bind, 
            params={"symbol": symbol, "lookback": lookback},
            parse_dates=['timestamp']
        )
        
        # Data comes in descending order, so we must reverse it
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Fetched {len(df)} rows of combined feature data.")
        return df

    def _prepare_inputs(self, df: pd.DataFrame) -> (Dict[str, torch.Tensor], pd.DataFrame):
        """
        Takes the raw dataframe and processes it into the two
        required inputs:
        1. x_blocks (for TFT/TCN)
        2. x_tabular (for XGBoost)
        """
        # --- 1. Create Tabular Features (for XGB) ---
        tabular_df = create_tabular_features(df, TABULAR_FEATURE_COLS, ROLL_WINDOWS)
        
        # Get the *very last row* of tabular features
        x_tabular_row_df = tabular_df.iloc[[-1]]
        
        # --- 2. Create Sequential Features (for TFT/TCN) ---
        # Get the *last SEQ_LEN* rows of sequential features
        x_seq_df = pd.DataFrame()
        for bname, cols in FEATURE_BLOCKS.items():
            x_seq_df = pd.concat([x_seq_df, df[cols]], axis=1)
            
        x_seq_raw = x_seq_df.iloc[-SEQ_LEN:].values.astype(np.float32)
        
        # Apply the scaler
        x_seq_scaled = (x_seq_raw - self.scaler_mean) / self.scaler_std
        
        # Create the x_blocks dictionary
        x_blocks = {}
        start_col = 0
        for bname, cols in FEATURE_BLOCKS.items():
            end_col = start_col + len(cols)
            block_data = x_seq_scaled[:, start_col:end_col]
            
            # Add batch dimension [1, L, F] and move to device
            x_blocks[bname] = torch.from_numpy(block_data).float().unsqueeze(0).to(DEVICE)
            start_col = end_col
            
        return x_blocks, x_tabular_row_df

    def predict(self, symbol: str) -> Dict[str, float]:
        """
        Runs the full end-to-end hybrid inference pipeline.
        """
        if not self.is_ready:
            raise RuntimeError("InferenceService is not ready. Check model loading logs.")
            
        db = SessionLocal()
        try:
            # 1. Fetch Data
            # Fetch enough data for seq_len + rolling windows
            total_lookback = SEQ_LEN + max(ROLL_WINDOWS) + 5
            raw_df = self._fetch_inference_data(db, symbol, total_lookback)
            
            if len(raw_df) < total_lookback:
                raise ValueError(f"Not enough data for inference on {symbol}. Need {total_lookback}, got {len(raw_df)}")

            # 2. Prepare Inputs
            x_blocks, x_tabular = self._prepare_inputs(raw_df)

            # 3. Run Inference (all models)
            with torch.no_grad():
                # PyTorch Models
                preds_tft = self.models['tft'](x_blocks)
                preds_tcn = self.models['tcn'](x_blocks)
                
                # XGB Models
                pred_xgb_price = _xgb_predict(self.models['xgb_price'], x_tabular)
                pred_xgb_vol = _xgb_predict(self.models['xgb_vol'], x_tabular)

                # --- 4. Stack and Blend ---
                
                # Stack price predictions: [1, 3]
                stacked_price = torch.stack([
                    preds_tft['price'][0],
                    preds_tcn['price'][0],
                    torch.tensor(pred_xgb_price, device=DEVICE)
                ], dim=0).unsqueeze(0)
                
                # Stack volatility predictions: [1, 3]
                stacked_vol = torch.stack([
                    preds_tft['vol'][0],
                    preds_tcn['vol'][0],
                    torch.tensor(pred_xgb_vol, device=DEVICE)
                ], dim=0).unsqueeze(0)

                # --- 5. Get Final Predictions ---
                final_price = self.models['blender_price'](stacked_price).item()
                final_vol = self.models['blender_vol'](stacked_vol).item()
                
                # Get TFT feature weights for interpretability
                # (Averaging over the sequence length)
                tft_weights = preds_tft['feature_weights'].mean(dim=(0,1)).cpu().numpy()
                feature_importance = {name: float(w) for name, w in zip(FEATURE_BLOCKS.keys(), tft_weights)}

            return {
                "price_prediction": final_price,
                "volatility_prediction": final_vol,
                "feature_importance": feature_importance,
                "tft_price": preds_tft['price'].item(),
                "tcn_price": preds_tcn['price'].item(),
                "xgb_price": pred_xgb_price,
            }

        except Exception as e:
            self.logger.error(f"Error during inference for {symbol}: {e}", exc_info=True)
            raise e
        finally:
            db.close()

# --- Global Singleton Instance ---
# This line creates one instance of the service that will be
# loaded on app startup and shared across all API requests.
try:
    inference_service = HybridInferenceService()
except Exception as e:
    inference_service = None
    logging.critical(f"Failed to initialize InferenceService on startup: {e}")
def build_decision(self, ctx: MarketContext) -> HybridDecision:
        """
        Use existing TFT + TCN + XGB models to produce a trade-ready hybrid decision.
        Does NOT change how models work; only interprets them.
        """
        if not self.is_ready:
            raise RuntimeError("InferenceService not ready.")

        db = SessionLocal()
        try:
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
                    debug={"reason": "insufficient_data", "rows": len(raw_df)},
                )

            # reuse prep function
            x_blocks, x_tabular = self._prepare_inputs(raw_df)

            with torch.no_grad():
                preds_tft = self.models["tft"](x_blocks)
                preds_tcn = self.models["tcn"](x_blocks)
                xgb_price = _xgb_predict(self.models["xgb_price"], x_tabular) if self.models.get("xgb_price") else None
                xgb_vol = _xgb_predict(self.models["xgb_vol"], x_tabular) if self.models.get("xgb_vol") else None

            expert = ExpertSignals(
                tft_price=float(preds_tft["price"].item()),
                tcn_price=float(preds_tcn["price"].item()),
                xgb_price=xgb_price,
                tft_vol=float(preds_tft["vol"].item()),
                tcn_vol=float(preds_tcn["vol"].item()),
                xgb_vol=xgb_vol,
            )

            # simple realized vol + trend for context
            ret = raw_df["close"].pct_change()
            rv_24h = float(ret.rolling(96).std().iloc[-1] or 0.0)
            trend_score = float(ret.rolling(48).mean().iloc[-1] or 0.0)

            features = {
                "rv_24h": rv_24h,
                "funding_1h": 0.0,       # plug real funding when available
                "trend_score": trend_score,
            }

            # 1) meta ensemble: edge + raw dir
            meta = meta_predict(features, expert, ctx)

            # 2) meta-label: execute? how big?
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
                    debug={"expert": expert.dict(), "meta": meta},
                )

            # 3) bandit: weights + strategy tag
            weights, tag = bandit_weights(features, expert, meta, ctx)

            # For now: use meta's direction as final; weights are for debugging / future extensions
            direction = meta["dir_raw"]

            return HybridDecision(
                symbol=ctx.symbol,
                instrument_type=ctx.instrument_type,
                direction=direction,
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
            self.logger.error(f"Hybrid decision failed for {ctx.symbol}: {e}", exc_info=True)
            # Fail safe: no trade
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