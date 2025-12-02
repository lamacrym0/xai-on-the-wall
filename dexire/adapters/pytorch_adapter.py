import torch
import torch.nn as nn
import numpy as np
from typing import List, Union

from .model_adapter import AbstractModelAdapter


class PyTorchModelAdapter(AbstractModelAdapter):
    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            return outputs.detach().cpu().numpy()

    def get_layer_names(self) -> List[str]:
        layer_names = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                layer_names.append(name)
        return layer_names

    def get_layer_output(self, X: np.ndarray, layer_idx: Union[int, str]) -> np.ndarray:
        x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        if isinstance(layer_idx, int):
            layers = list(self.model.children())  # <–– PATCH QUI
            if layer_idx < 0:
                layer_idx = len(layers) + layer_idx
            target_layer = layers[layer_idx]
        elif isinstance(layer_idx, str):
            target_layer = dict(self.model.named_modules()).get(layer_idx)
            if target_layer is None:
                raise ValueError(f"Layer '{layer_idx}' not found in model.")
        else:
            raise ValueError("layer_idx must be int or str")

        activation_container = {"output": None}

        def hook(module, input, output):
            activation_container["output"] = output.detach()

        hook_handle = target_layer.register_forward_hook(hook)

        with torch.no_grad():
            _ = self.model(x_tensor)

        hook_handle.remove()

        if activation_container["output"] is None:
            raise RuntimeError(f"No activation captured for layer {layer_idx}")

        return activation_container["output"].cpu().numpy()

    def get_candidate_layer_indices(self) -> List[int]:
        return [i for i, layer in enumerate(self.model.children()) if isinstance(layer, nn.Linear)]

    def get_num_classes(self) -> int:
        dummy_input = torch.randn(1, next(self.model.parameters()).shape[1]).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        return output.shape[-1]
