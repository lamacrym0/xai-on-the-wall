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

    if "target_labels" in data:
        unique_labels = data["target_labels"]
    else:
        unique_labels = np.unique(data["y_train"])
    
    dynamic_class_names = [str(label) for label in unique_labels]
    
    if not dynamic_class_names:
        dynamic_class_names = ["Class0", "Class1"]

    dexire = DEXiRE(
        model=adapter,
        feature_names=safe_names,
        class_names=dynamic_class_names,
    )

    rule_set = dexire.extract_rules(data["X_train"], data["y_train"])

    rules_str = str(rule_set)
    feature_means = {}
    feature_counts = {}
    for rule in rule_set.rules:
        for feature_rule in str(rule).split("("):
            if len(feature_rule.split(" ")) >= 3:
                name = feature_rule.split(" ")[0]
                value = abs(float(feature_rule.split(" ")[2].split(")")[0]))
                try:
                    feature_counts[name] += 1
                    feature_means[name] = feature_means[name] + value
                except:
                    feature_counts[name] = 1
                    feature_means[name] = value
    for feature,value in feature_means.items():
        value = value / feature_counts[feature]
        feature_means[feature] = value

    for safe, original in mapping.items():
        rules_str = rules_str.replace(safe, original)
    
    sorted_means = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)
    sorted_counts = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    count_length=0
    for rule in rule_set.rules:
            count_length += (len(rule)-1)
    avg_length = count_length / len(rule_set.rules) if len(rule_set.rules)>0 else 0

    return rules_str, sorted_counts, sorted_means, len(rule_set.rules),avg_length