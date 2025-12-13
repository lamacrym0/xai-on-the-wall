from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd

def load_data(dataset):
    match dataset:
        case "Iris":
            X, y = load_iris(return_X_y=True)
            feature_names = load_iris().feature_names
        case "Wine":
            X, y = load_wine(return_X_y=True)
            feature_names = load_wine().feature_names
        case "Breast Cancer":
            X, y = load_breast_cancer(return_X_y=True)
            feature_names = load_breast_cancer().feature_names
        case _:
            dataset = pd.read_csv(dataset)
            target_name = dataset.columns[-1]
            if target_name not in dataset.columns:
                raise KeyError(f"Target column '{target_name}' not found in DataFrame columns: {list(dataset.columns)}")
            y = dataset[target_name].copy()
            X = dataset.drop(columns=[target_name]).copy()
            feature_names = X.columns.tolist()
            X = X.to_numpy()
    return X, y, feature_names