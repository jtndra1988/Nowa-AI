# inference_tst.py

import os
import json
import torch
import numpy as np

from app.ml.adv.feature_engineering import FeatureEngineering
from app.ml.adv.models_tst import TimeSeriesTransformer


class TSTInferenceService:
    def __init__(self, artifact_dir: str, device: torch.device = None):
        self.model, self.metadata = TimeSeriesTransformer.load_from_artifact(artifact_dir, device=device)
        self.feature_list = self.metadata["feature_list"]
        self.prediction_length = self.metadata["prediction_length"]
        self.target_size = self.metadata["target_size"]
        self.device = device or torch.device("cpu")

        # If saved: load scaler/encoder files
        # e.g., self.scaler = load_pickle(os.path.join(artifact_dir, "scaler.pkl"))

    def preprocess(self, raw_input: dict) -> torch.Tensor:
        """
        raw_input: dictionary (or DataFrame row) of raw market data
        Returns: tensor of shape (1, context_length, num_features)
        """
        fe = FeatureEngineering()
        df_features = fe.transform(raw_input)
        if set(df_features.columns.tolist()) != set(self.feature_list):
            raise ValueError(f"Feature mismatch: expected {self.feature_list} but got {df_features.columns.tolist()}")

        arr = df_features[self.feature_list].values.astype(np.float32)
        tensor = torch.from_numpy(arr).to(self.device)
        tensor = tensor.unsqueeze(0)  # batch dimension
        return tensor

    def predict(self, raw_input: dict) -> np.ndarray:
        tensor = self.preprocess(raw_input)
        with torch.no_grad():
            output = self.model(tensor)
        # output is tensor of shape (batch_size=1, prediction_length, target_size)
        result = output.cpu().numpy()
        return result

if __name__ == "__main__":
    artifact_dir = "models/tst/v1.0_20251120T123456Z"  # update accordingly
    service = TSTInferenceService(artifact_dir)
    sample_input = {...}  # fill with appropriate raw input
    preds = service.predict(sample_input)
    print("Prediction shape:", preds.shape)  # should be (1, prediction_length, target_size)
    print("Predictions:", preds)
