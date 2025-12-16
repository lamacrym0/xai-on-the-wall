import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from src.model_builder import build_mlp_from_layer_df
from src.train import train, prepare_data

def test_build_mlp_structure():
    input_size = 10
    layer_config = pd.DataFrame([
        {"units": 32, "activation": "relu"},
        {"units": 16, "activation": "sigmoid"}
    ])
    
    model = build_mlp_from_layer_df(input_size, layer_config, output_size=1)
    
    assert isinstance(model, nn.Sequential)
    assert len(model) == 6 
    assert isinstance(model[0], nn.Linear)
    assert model[0].in_features == 10
    assert model[0].out_features == 32
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[5], nn.Sigmoid) 

def test_prepare_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100) 
    
    data = prepare_data(X, y) #
    
    assert "X_train_tensor" in data
    assert "y_test_tensor" in data
    assert data["input_size"] == 5
    assert not data["is_multiclass"]
    assert torch.is_tensor(data["X_train_tensor"])

def test_train_loop():
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, 50)
    
    model = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
    
    trained_model, data, history = train(
        X, y, epochs=2, lr=0.01, model=model, use_wandb=False
    ) 
    
    assert trained_model is not None
    assert len(history["train_loss"]) == 2
    assert "val_acc" in history