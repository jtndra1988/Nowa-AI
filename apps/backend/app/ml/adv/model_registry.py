import logging
import os
from typing import Any, Dict, Optional

from app.ml.adv.models_tft import TFTPredictor
from app.ml.adv.models_tcn import TCNPredictor
from app.ml.adv.models_tst import TSTPredictor
from app.ml.adv.inference_xgb import XGBInferenceService
from app.ml.adv.options_vol_model import OptionsVolFeatureIngestion
from app.ml.adv.macro_onchain_model import MacroOnchainFeatureIngestion
from app.ml.adv.llm_narrative_model import llm_engine
from app.ml.adv.rl_execution_agent import rl_agent

logger = logging.getLogger(__name__)

# Optional: environment-driven expected versions for hard checks
EXPECTED_VERSIONS: Dict[str, Optional[str]] = {
    "tft": os.getenv("NOWA_TFT_VERSION"),
    "tcn": os.getenv("NOWA_TCN_VERSION"),
    "tst": os.getenv("NOWA_TST_VERSION"),
    "xgb": os.getenv("NOWA_XGB_VERSION"),
    "llm": os.getenv("NOWA_LLM_VERSION"),
    "rl": os.getenv("NOWA_RL_VERSION"),
    "options_ingestor": os.getenv("NOWA_OPT_VERSION"),
    "macro_ingestor": os.getenv("NOWA_MACRO_VERSION"),
    "ensemble": os.getenv("NOWA_ENS_VERSION"),   # optional
}



class ModelRegistry:
    """
    Central registry for all core models & ingestors.

    Responsibilities:
      - Load each model/ingestor ONCE per process.
      - Cache instances for reuse.
      - Extract metadata (including version) if available.
      - Optionally enforce expected versions via env vars.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        logger.info("[ModelRegistry] Initializing global model registry...")
        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model(self, name: str):
        if name not in self._models:
            self._load_model(name)
        return self._models.get(name)
    def _load_model(self, name: str):
        if name == "ensemble":
           from app.ml.ensemble import EnsembleStackerService
           obj = EnsembleStackerService()
           self._register(name, obj)

    def get_metadata(self, key: str) -> Dict[str, Any]:
        return self._metadata.get(key, {})

    def all_models(self) -> Dict[str, Any]:
        return dict(self._models)

    def all_metadata(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Internal loading logic
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._register("tft", self._load_tft())
        self._register("tcn", self._load_tcn())
        self._register("tst", self._load_tst())
        self._register("xgb", self._load_xgb())
        self._register("options_ingestor", self._load_options_ingestor())
        self._register("macro_ingestor", self._load_macro_ingestor())
        self._register("llm", self._load_llm())
        self._register("rl", self._load_rl())

        logger.info(
            "[ModelRegistry] Loaded models: %s",
            ", ".join(sorted(self._models.keys())) or "(none)",
        )

    def _register(self, key: str, obj: Optional[Any]) -> None:
        if obj is None:
            logger.warning("[ModelRegistry] Model '%s' is not available.", key)
            return

        self._models[key] = obj
        meta = self._extract_metadata(key, obj)
        self._metadata[key] = meta
        self._check_version(key, meta)

    # --- individual loaders ---

    def _load_tft(self) -> Optional[Any]:
        try:
            model = TFTPredictor()
            if hasattr(model, "is_model_loaded") and not model.is_model_loaded():
                logger.warning("[ModelRegistry] TFTPredictor not loaded; skipping.")
                return None
            return model
        except Exception as e:  # noqa: BLE001
            logger.error("[ModelRegistry] Failed to load TFTPredictor: %s", e, exc_info=True)
            return None

    def _load_tcn(self) -> Optional[Any]:
        try:
            model = TCNPredictor()
            if hasattr(model, "is_model_loaded") and not model.is_model_loaded():
                logger.warning("[ModelRegistry] TCNPredictor not loaded; skipping.")
                return None
            return model
        except Exception as e:  # noqa: BLE001
            logger.error("[ModelRegistry] Failed to load TCNPredictor: %s", e, exc_info=True)
            return None

    def _load_tst(self) -> Optional[Any]:
        try:
            model = TSTPredictor()
            if hasattr(model, "is_model_loaded") and not model.is_model_loaded():
                logger.warning("[ModelRegistry] TSTPredictor not loaded; skipping.")
                return None
            return model
        except Exception as e:  # noqa: BLE001
            logger.error("[ModelRegistry] Failed to load TSTPredictor: %s", e, exc_info=True)
            return None

    def _load_xgb(self) -> Optional[Any]:
        try:
            svc = XGBInferenceService()
            if not getattr(svc, "is_ready", False):
                logger.warning("[ModelRegistry] XGBInferenceService not ready; skipping.")
                return None
            return svc
        except Exception as e:  # noqa: BLE001
            logger.error("[ModelRegistry] Failed to init XGBInferenceService: %s", e, exc_info=True)
            return None

    def _load_options_ingestor(self) -> Optional[Any]:
        try:
            return OptionsVolFeatureIngestion()
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[ModelRegistry] Failed to init OptionsVolFeatureIngestion: %s",
                e,
                exc_info=True,
            )
            return None

    def _load_macro_ingestor(self) -> Optional[Any]:
        try:
            return MacroOnchainFeatureIngestion()
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[ModelRegistry] Failed to init MacroOnchainFeatureIngestion: %s",
                e,
                exc_info=True,
            )
            return None

    def _load_llm(self) -> Optional[Any]:
        """
        llm_engine is already a singleton; just return it if it's usable.
        """
        try:
            if hasattr(llm_engine, "is_model_loaded") and not llm_engine.is_model_loaded():
                logger.warning("[ModelRegistry] LLM engine reports not loaded.")
            return llm_engine
        except Exception as e:  # noqa: BLE001
            logger.error("[ModelRegistry] Error validating llm_engine: %s", e, exc_info=True)
            return None

    def _load_rl(self) -> Optional[Any]:
        """
        rl_agent is a singleton RLAgent; ensure it is initialized.
        """
        try:
            # Access is_model_loaded to force initialization failure to surface
            _ = getattr(rl_agent, "is_model_loaded", lambda: False)()
            return rl_agent
        except Exception as e:  # noqa: BLE001
            logger.error("[ModelRegistry] Error initializing rl_agent: %s", e, exc_info=True)
            return None

    # --- metadata & version checks ---

    def _extract_metadata(self, key: str, obj: Any) -> Dict[str, Any]:
        """
        Best-effort metadata extraction:
          - Prefer obj.get_metadata() if it exists.
          - Fallback to known methods (e.g. rl.get_policy_version()).
          - Fallback to empty dict if nothing available.
        """
        # 1) Generic get_metadata()
        get_meta = getattr(obj, "get_metadata", None)
        if callable(get_meta):
            try:
                meta = get_meta()
                if isinstance(meta, dict):
                    return meta
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[ModelRegistry] %s.get_metadata() failed: %s", key, e, exc_info=True
                )

        meta: Dict[str, Any] = {"source": key}

        # 2) RL policy version
        if key == "rl":
            try:
                ver = getattr(obj, "get_policy_version", lambda: "unknown")()
                meta["version"] = ver
            except Exception:
                pass

        # 3) LLM basic status
        if key == "llm":
            try:
                loaded = getattr(obj, "is_model_loaded", lambda: False)()
                meta["loaded"] = bool(loaded)
            except Exception:
                pass

        return meta

    def _check_version(self, key: str, meta: Dict[str, Any]) -> None:
        """
        Compare model version against EXPECTED_VERSIONS (if set). If mismatch,
        log an error and remove the model from the registry.
        """
        expected = EXPECTED_VERSIONS.get(key)
        if not expected:
            return

        actual = str(meta.get("version", "unknown"))
        if actual == "unknown":
            logger.warning(
                "[ModelRegistry] Model '%s' has no version in metadata; "
                "cannot validate against expected=%s",
                key,
                expected,
            )
            return

        if actual != expected:
            logger.error(
                "[ModelRegistry] VERSION MISMATCH for '%s': expected=%s, got=%s. "
                "This model will be DISABLED.",
                key,
                expected,
                actual,
            )
            # Disable model on hard mismatch
            self._models.pop(key, None)


# Global singleton used across backend
model_registry = ModelRegistry()
