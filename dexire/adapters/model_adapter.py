import numpy as np
from typing import List, Union

class AbstractModelAdapter:
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_layer_names(self) -> List[str]:
        raise NotImplementedError

    def get_layer_output(self, X: np.ndarray, layer_identifier: Union[int, str]) -> np.ndarray:
        raise NotImplementedError

    def get_candidate_layer_indices(self):
        raise NotImplementedError

    def is_classification(self) -> bool:
        raise NotImplementedError