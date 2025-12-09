
from dexire.dexire import DEXiRE
from dexire.adapters.pytorch_adapter import PyTorchModelAdapter
import numpy as np

def get_dexire_rules(model, data,feature_names):
    adapter = PyTorchModelAdapter(model)
    dexire = DEXiRE(model=adapter, class_names=['malignant', 'benign'])
    print("y_train unique values:", np.unique(data["y_train"]))
    print("class_names:", dexire.class_names)
    return dexire.extract_rules(data["X_train"], data["y_train"])