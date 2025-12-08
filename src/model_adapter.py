import numpy as np
import pandas as pd
import torch


class ModelAdapter:
    """
    Adapter for a single-output sigmoid PyTorch model.
    Exposes .predict(X) -> 0/1 labels.
    """

    def __init__(self, model, device=None, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()

    def _to_tensor(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        return torch.from_numpy(X_np).to(self.device)

    def predict(self, X):
        """
        Returns binary labels (0/1) by thresholding the model's sigmoid output.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            y = self.model(X_tensor)      
            y = y.view(-1)                  

        probs = y.detach().cpu().numpy()
        labels = (probs >= self.threshold).astype(int)  # 0 or 1

        return labels
