from dexire_exo.ga_engine import GAEngine
from dexire_exo.rule_formatter import format_if_elif_else
from sklearn.metrics import accuracy_score, classification_report
from src.config import Config
from src.model_adapter import ModelAdapter

def get_dexire_exo_rules(feature_names, model, data):
    config = Config("config.yaml")

    model_adapter = ModelAdapter(model)

    engine = GAEngine(
        model_adapter=model_adapter,
        feature_names=feature_names,
        config=config
    )

    X_train = data["X_train"]   
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    best = engine.evolve(X_train, y_train)
    y_pred_te, uncov_te = engine.predict_rules(best, X_test)

    mask = y_pred_te != -1
    test_acc = accuracy_score(y_test[mask], y_pred_te[mask]) if mask.any() else 0.0

    return best, test_acc, uncov_te, engine
