import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.model(x)
        return x
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

def build_model(input_size):

    return NN(input_size=input_size)