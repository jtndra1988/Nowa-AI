# inference_tcn.py
import os
import json
import torch
import numpy as np
from app.ml.adv.models_tcn import TemporalConvolutionalNetwork
from app.ml.adv.feature_engineering import FEATURE_CONFIG, process_market_data

class TCNInferenceService:
    def __init__(self, artifact_dir: str, device: torch.device = None):
        self.model, self.metadata = TemporalConvolutionalNetwork.load_from_artifact(artifact_dir, device=device)
        self.feature_list = self.metadata["feature_list"]
        self.device = device or torch.device("cpu")
        # If you previously saved scaler/encoder, load them here
        # Example:
        # self.scaler = load_pickle(os.path.join(artifact_dir, "scaler.pkl"))
        # self.encoder = load_pickle(os.path.join(artifact_dir, "encoder.pkl"))

    def preprocess(self, raw_input: dict) -> torch.Tensor:
        """
        raw_input: dictionary or DataFrame row containing raw market features
        Returns a torch.Tensor shaped for model input.
        """
        # Use the same feature engineering logic as training
        fe = FeatureEngineering()
        df_features = fe.transform(raw_input)  # returns DataFrame with all features
        # Validate feature list
        if set(df_features.columns.tolist()) != set(self.feature_list):
            raise ValueError(f"Feature mismatch: expected {self.feature_list} but got {df_features.columns.tolist()}")

        # Convert to tensor and send to device
        arr = df_features[self.feature_list].values.astype(np.float32)
        tensor = torch.from_numpy(arr).to(self.device)
        tensor = tensor.unsqueeze(0)  # add batch dimension if needed
        return tensor

    def predict(self, raw_input: dict) -> np.ndarray:
        tensor = self.preprocess(raw_input)
        with torch.no_grad():
             outputs = self.model(tensor)   # {"price": Tensor[1], "vol": Tensor[1]}
 
        price = outputs.get("price")
        if price is not None:
           return price.cpu().numpy()
        vol = outputs.get("vol")

        # e.g., return only price as a float or 1D numpy array
        if price is not None:
         return price.cpu().numpy()

        # fallback
        return np.zeros((1,), dtype=np.float32)

if __name__ == "__main__":
    # Example usage
    artifact_dir = "models/tcn/v1.0_20251120T123456Z"  # update accordingly
    service = TCNInferenceService(artifact_dir)
    sample_input = {...}  # fill with appropriate raw input fields
    prediction = service.predict(sample_input)
    print("Prediction:", prediction)
