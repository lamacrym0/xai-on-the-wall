# ─────────────────────────────────────────────
# Model builder from layer config
# ─────────────────────────────────────────────
import torch.nn as nn
import pandas as pd

def build_mlp_from_layer_df(input_size: int, df) -> nn.Module:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=["units", "activation"])

    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }

    layers = []
    prev = input_size

    for _, row in df.iterrows():
        try:
            units = int(row["units"])
        except (ValueError, TypeError):
            continue
        if units <= 0:
            continue

        act_name = str(row["activation"]).strip().lower() or "relu"
        if act_name not in act_map:
            act_name = "relu"

        layers.append(nn.Linear(prev, units))
        layers.append(act_map[act_name]())
        prev = units

    # Fallback if user emptied everything
    if not layers:
        layers = [
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        ]
        prev = 8

    layers.append(nn.Linear(prev, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)
