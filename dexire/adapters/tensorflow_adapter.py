from dexire.adapters.model_adapter import AbstractModelAdapter

import keras
import numpy as np
from typing import List

class TensorFlowModelAdapter(AbstractModelAdapter):
    def __init__(self, model: keras.Model):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_layer_names(self) -> List[str]:
        return [layer.name for layer in self.model.layers]

    def get_layer_output(self, X: np.ndarray, layer_idx: int) -> np.ndarray:
        intermediate_model = keras.Model(inputs=self.model.input, outputs=self.model.layers[layer_idx].output)
        return intermediate_model.predict(X)

    def get_candidate_layer_indices(self) -> List[int]:
        # Consider only Dense layers, excluding the last one (assumed to be output)
        dense_layers = [i for i, layer in enumerate(self.model.layers) if isinstance(layer, keras.layers.Dense)]
        if len(dense_layers) > 1:
            # Exclude last Dense (likely output layer)
            return dense_layers[:-1]
        return dense_layers

    def is_classification(self) -> bool:
        return self.model.output_shape[-1] > 1