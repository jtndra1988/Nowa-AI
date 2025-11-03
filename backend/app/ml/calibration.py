# unchanged, kept your temperature + platt utilities

import json
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

@dataclass
class PlattModel:
    # binary or one-vs-rest per class on max-logit
    coef_: np.ndarray
    intercept_: float

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        # For 3-class, calibrate max-logit confidence; keep argmax label unchanged.
        z = (logits.max(axis=1) * self.coef_[0]) + self.intercept_
        p = 1.0 / (1.0 + np.exp(-z))
        # Re-scale softmax by calibrated peak probability
        sm = softmax_rows(logits)
        peak = sm.max(axis=1, keepdims=True)
        scale = np.clip(p.reshape(-1,1) / (peak + 1e-9), 0.1, 10.0)
        return normalize_rows(sm * scale)

def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=1, keepdims=True) + 1e-9
    return x / s

def fit_platt(logits_val: np.ndarray, y_val: np.ndarray) -> PlattModel:
    # Calibrate max-logit to "correct-or-not" (1 if argmax==y)
    pred = logits_val.argmax(axis=1)
    correct = (pred == y_val).astype(int)
    z = logits_val.max(axis=1).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(z, correct)
    model = PlattModel(coef_=lr.coef_.copy(), intercept_=float(lr.intercept_[0]))
    return model

def save_platt(model: PlattModel, path: str):
    with open(path, "w") as f:
        json.dump({"coef": model.coef_.tolist(), "intercept": model.intercept_}, f)

def load_platt(path: str) -> PlattModel:
    js = json.load(open(path))
    return PlattModel(coef_=np.array(js["coef"]), intercept_=float(js["intercept"]))

def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray, lr: float = 0.01, steps: int = 200) -> float:
    """
    Temperature scaling via simple gradient descent on NLL.
    """
    T = 1.0
    for _ in range(steps):
        sm = softmax_rows(logits_val / T)
        # NLL gradient w.r.t 1/T -> chain rule for T
        idx = np.arange(len(y_val))
        nll = -np.log(sm[idx, y_val] + 1e-9).mean()
        # gradient approximate: dNLL/dT ≈ (sum_i ( (p_i - y_i_onehot) * logits_i ) ) / T^2
        onehot = np.zeros_like(sm); onehot[idx, y_val] = 1.0
        grad_T = ((sm - onehot) * logits_val / (T**2)).sum(axis=1).mean()
        T -= lr * grad_T
        T = float(np.clip(T, 0.5, 5.0))
    return T

def save_temperature(T: float, path: str):
    with open(path, "w") as f:
        json.dump({"temperature": T}, f)

def load_temperature(path: str) -> float:
    return float(json.load(open(path))["temperature"])

def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return softmax_rows(logits / max(T, 1e-6))
