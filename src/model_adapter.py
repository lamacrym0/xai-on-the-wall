import numpy as np
import pandas as pd
import torch

class ModelAdapter:
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
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            outputs = self.model(X_tensor)
            
            if outputs.shape[1] > 1:
                _, predicted = torch.max(outputs, 1)
                labels = predicted.cpu().numpy()
            else:
                outputs = outputs.view(-1)
                probs = outputs.detach().cpu().numpy()
                labels = (probs >= self.threshold).astype(int)

        return labels