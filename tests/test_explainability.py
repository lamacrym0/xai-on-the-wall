import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from explainer.dexire import get_dexire_rules
from explainer.ciu import get_explainer_CIU, get_ciu_instance

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

@pytest.fixture
def mock_data_model():
    model = SimpleModel()
    data = {
        "X_train": np.random.rand(20, 4),
        "y_train": np.random.randint(0, 2, 20),
        "target_labels": [0, 1]
    }
    feature_names = ["sepal_L", "sepal_W", "petal_L", "petal_W"]
    return model, data, feature_names

@patch("explainer.dexire.DEXiRE") 
def test_get_dexire_rules(mock_dexire_cls, mock_data_model):
    model, data, features = mock_data_model
    
    mock_instance = mock_dexire_cls.return_value
    mock_instance.extract_rules.return_value = ["IF feature1 > 0.5 THEN Class1"]
    mock_instance.class_names = ["Class0", "Class1"]
    
    rules_str, sorted_counts = get_dexire_rules(model, data, features)
    
    assert "IF feature1 > 0.5 THEN Class1" in rules_str
    assert isinstance(sorted_counts, list) 
    mock_dexire_cls.assert_called_once() 

@patch("explainer.ciu.ciu.CIU")
def test_ciu_wrapper(mock_ciu_cls, mock_data_model):
    model, data, features = mock_data_model
    output_names = ["Class0", "Class1"]
    
    explainer = get_explainer_CIU(model, data, output_names, features) 
    mock_ciu_cls.assert_called_once()
    
    instance = pd.DataFrame([data["X_train"][0]], columns=features)
    get_ciu_instance(explainer, instance) 
    
    explainer.explain.assert_called_once()