from dexire.dexire import DEXiRE
from dexire.adapters.pytorch_adapter import PyTorchModelAdapter
import numpy as np
import re

def make_sympy_safe_names(feature_names):
    safe_names = []
    mapping = {}

    for i, name in enumerate(feature_names):
        name_str = str(name)

        safe = re.sub(r"[^0-9a-zA-Z_]", "_", name_str)

        if not safe or safe[0].isdigit():
            safe = f"f{i}_{safe}"

        safe_names.append(safe)
        mapping[safe] = name_str  

    return safe_names, mapping


def get_dexire_rules(model, data, feature_names):
    safe_names, mapping = make_sympy_safe_names(feature_names)

    adapter = PyTorchModelAdapter(model)

    dexire = DEXiRE(
        model=adapter,
        feature_names=safe_names,
        class_names=["malignant", "benign"],
    )

    print("y_train unique values:", np.unique(data["y_train"]))
    print("class_names:", dexire.class_names)

    rule_set = dexire.extract_rules(data["X_train"], data["y_train"])

    rules_str = str(rule_set)

    for safe, original in mapping.items():
        rules_str = rules_str.replace(safe, original)

    return rules_str
