import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from src.preprocess import preprocess_data

project = "xai-on-the-wall"

def prepare_data(X, y, test_split=0.2, val_split=0.2, random_seed=42):
    """
    Prepares data for training: splits, preprocessing, and tensor conversion.
    Automatically handles formatting for Binary vs Multiclass targets.
    """
    # Set seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Auto-detect classification mode
    classes = np.unique(y)
    n_classes = len(classes)
    is_multiclass = n_classes > 2
    
    X = preprocess_data(X)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def to_tensor(d): 
        return torch.tensor(d, dtype=torch.float32).to(device)

    def to_label(d):
        if is_multiclass:
            return torch.tensor(d, dtype=torch.long).to(device)
        else:
            return torch.tensor(d, dtype=torch.float32).view(-1, 1).to(device)

    data = {
        "X_train_tensor": to_tensor(X_train), "y_train_tensor": to_label(y_train),
        "X_val_tensor": to_tensor(X_val),     "y_val_tensor": to_label(y_val),
        "X_test_tensor": to_tensor(X_test),   "y_test_tensor": to_label(y_test),
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "input_size": X_train.shape[1],
        "target_labels": classes,
        "is_multiclass": is_multiclass
    }
    return data

def train(X, y, epochs=50, lr=0.001, model=None, random_seed=42, 
          optimizer_name="Adam", loss_name="Auto", 
          use_wandb=False):
    """
    Main training loop.
    Supports dynamic Loss selection and Optimizer selection.
    """
    
    # 1. Prepare Data
    data = prepare_data(X, y, random_seed=random_seed)
    is_multiclass = data["is_multiclass"]
    num_classes = len(data["target_labels"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        raise ValueError("Error: No model provided to train() function.")

    model.to(device)
    
    # 2. Auto Loss Selection
    if loss_name == "Auto":
        if is_multiclass:
            criterion = nn.CrossEntropyLoss()
            loss_name = "CrossEntropy (Auto)"
        else:
            criterion = nn.BCELoss()
            loss_name = "BCELoss (Auto)"
    else:
        # User manual
        if loss_name == "CrossEntropyLoss": criterion = nn.CrossEntropyLoss()
        elif loss_name == "MSELoss": criterion = nn.MSELoss()
        elif loss_name == "L1Loss": criterion = nn.L1Loss()
        else: criterion = nn.BCELoss()

    # 3. Optimizer Selection
    if optimizer_name == "SGD": optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop": optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else: optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epochs": []}
    
    run = None
    if use_wandb:
        wandb.login()
        run = wandb.init(project=project, config={"epochs": epochs, "lr": lr})

    # 4. Hybrid Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data["X_train_tensor"])
        
        loss = criterion(output, data["y_train_tensor"])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Adaptive Accuracy Calculation
            if is_multiclass:
                # Multiclass: use argmax to get class index
                train_preds = torch.argmax(output, dim=1)
                train_acc = (train_preds == data["y_train_tensor"]).float().mean()
            else:
                # Binary: threshold probability at 0.5
                train_preds = (output > 0.5).float()
                train_acc = (train_preds == data["y_train_tensor"]).float().mean()

            # Validation phase
            model.eval()
            val_output = model(data["X_val_tensor"])
            val_loss = criterion(val_output, data["y_val_tensor"])
            
            if is_multiclass:
                val_preds = torch.argmax(val_output, dim=1)
                val_acc = (val_preds == data["y_val_tensor"]).float().mean()
            else:
                val_preds = (val_output > 0.5).float()
                val_acc = (val_preds == data["y_val_tensor"]).float().mean()

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc.item())
        history["val_loss"].append(val_loss.item())
        history["val_acc"].append(val_acc.item())
        
        if use_wandb and run:
             run.log({"Train Loss": loss.item(), "Train Accuracy": train_acc.item(), "Val Loss": val_loss.item(), "Val Accuracy": val_acc.item()})

    if run:
        run.finish()

    return model, data, history