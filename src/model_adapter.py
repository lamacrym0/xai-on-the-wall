import numpy as np
import pandas as pd
import torch


class ModelAdapter:
    """
    Make a PyTorch model look like an sklearn classifier for DexiRE-Exo:
    - .predict(X) -> class labels (0/1 for binary, 0..K-1 for multi-class)
    """

    def __init__(self, model, device=None):
        self.model = model
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
        Returns integer class labels (argmax over the second dim).
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            logits = self.model(X_tensor)          
            labels = torch.argmax(logits, dim=1)  

        return labels.detach().cpu().numpy()
