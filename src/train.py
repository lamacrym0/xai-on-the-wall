import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from config import NUM_EPOCHS, LEARNING_RATE, RANDOM_SEED, TEST_SPLIT, VAL_SPLIT
from preprocess import preprocess_data
from model import build_model

project = "xai-on-the-wall"


def prepare_data(X,y,test_split=TEST_SPLIT, val_split=VAL_SPLIT, random_seed=RANDOM_SEED):
    X = preprocess_data(X)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_seed, stratify=y_temp
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    input_size = X_train.shape[1]
    target_labels = np.unique(y_train)

    return {
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val_tensor,
        "y_val": y_val_tensor,
        "X_test_tensor": X_test_tensor,
        "y_test_tensor": y_test_tensor,
        "X_test": X_test,
        "y_test": y_test,
        "input_size": input_size,
        "target_labels": target_labels,
    }


def train(X,y,epochs=NUM_EPOCHS, lr=LEARNING_RATE, model=None):
    data = prepare_data(X, y)
    print(f"Target labels: {data['target_labels']}")
    if model is None:
        model = nn.Sequential(
            nn.Linear(data["input_size"], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    wandb.login()
    config = {"epochs": epochs, "lr": lr}

    with wandb.init(project=project, config=config) as run:
        print(f"lr: {config['lr']}")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        X_train = data["X_train_tensor"]
        y_train = data["y_train_tensor"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_preds = (output > 0.5).float()
                train_acc = (train_preds == y_train).float().mean()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
                val_preds = (val_output > 0.5).float()
                val_acc = (val_preds == y_val).float().mean()

            print(
                f"Epoch {epoch+1:02d} | "
                f"Loss: {loss.item():.4f} | Acc: {train_acc.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f}"
            )

            if run is not None:
                run.log(
                    {
                        "Train Loss": loss.item(),
                        "Train Accuracy": train_acc.item(),
                        "Val Loss": val_loss.item(),
                        "Val Accuracy": val_acc.item(),
                    }
                )
    return model, data