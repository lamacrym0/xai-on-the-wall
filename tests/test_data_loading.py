import pytest
import pandas as pd
import numpy as np
import os
from src.load_data import load_data

@pytest.mark.parametrize("dataset_name, expected_shape_x", [
    ("Iris", (150, 4)),
    ("Wine", (178, 13)),
    ("Breast Cancer", (569, 30))
])
def test_load_builtin_datasets(dataset_name, expected_shape_x):
    X, y, feature_names = load_data(dataset_name)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == expected_shape_x
    assert len(y) == expected_shape_x[0]
    assert len(feature_names) == expected_shape_x[1]

def test_load_csv_dataset(tmp_path):
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [0.1, 0.2, 0.3, 0.4],
        'target': [0, 1, 0, 1]
    })
    csv_file = tmp_path / "test_data.csv"
    df.to_csv(csv_file, index=False)

    X, y, feature_names = load_data(str(csv_file))

    assert X.shape == (4, 2)
    assert len(y) == 4
    assert feature_names == ['feature1', 'feature2']
    assert np.array_equal(y, df['target'].values)

def test_load_data_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")