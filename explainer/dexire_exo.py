from dexire_exo.ga_engine import GAEngine
from dexire_exo.rule_formatter import format_if_elif_else
from sklearn.metrics import accuracy_score, classification_report
from src.config import Config

def get_dexire_exo_rules(feature_names, model, data):
    config = Config("config.yaml")
    engine = GAEngine(model_adapter=model, feature_names=feature_names, config=config)
    best = engine.evolve(data["X_train"], data["y_train"])
    y_pred_te, uncov_te = engine.predict_rules(best, data["X_test"])
    mask = y_pred_te != -1
    test_acc = accuracy_score(data["y_test"][mask], y_pred_te[mask]) if mask.any() else 0.0
    return best, test_acc, uncov_te, engine