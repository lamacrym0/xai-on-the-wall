# src/model_builder.py
import torch.nn as nn
import pandas as pd

def build_mlp_from_layer_df(input_size: int, df, output_size: int = 1) -> nn.Module:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=["units", "activation"])

    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "sigmoid": nn.Sigmoid,
    }

    layers = []
    prev = input_size

    for _, row in df.iterrows():
        try:
            units = int(row["units"])
        except (ValueError, TypeError):
            continue
        if units <= 0: continue
        
        act_name = str(row["activation"]).strip().lower() or "relu"
        if act_name not in act_map: act_name = "relu"

        layers.append(nn.Linear(prev, units))
        layers.append(act_map[act_name]())
        prev = units
        
    if not layers:
        layers = [nn.Linear(input_size, 16), nn.ReLU()]
        prev = 16

    layers.append(nn.Linear(prev, output_size))

    if output_size == 1:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)