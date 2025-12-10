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
    Returns a dictionary containing both PyTorch tensors (for training) 
    and Numpy arrays (for XAI tools).
    """
    # Set seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Preprocessing
    X = preprocess_data(X)
    
    # Split: Train+Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed, stratify=y
    )
    
    # Split: Train vs Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_seed, stratify=y_temp
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Helper to convert to tensor on device
    def to_tensor(d): return torch.tensor(d, dtype=torch.float32).to(device)
    def to_label(d): return torch.tensor(d, dtype=torch.float32).view(-1, 1).to(device)

    data = {
        # Tensors for PyTorch training
        "X_train_tensor": to_tensor(X_train),
        "y_train_tensor": to_label(y_train),
        "X_val_tensor": to_tensor(X_val),
        "y_val_tensor": to_label(y_val),
        "X_test_tensor": to_tensor(X_test),
        "y_test_tensor": to_label(y_test),
        
        # Raw Numpy arrays for XAI (DexiRE, CIU, etc.)
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        
        # Metadata
        "input_size": X_train.shape[1],
        "target_labels": np.unique(y_train),
    }
    return data

def train(X, y, epochs=50, lr=0.001, model=None, random_seed=42, use_wandb=False, callback_progress=None):
    """
    Main training loop.
    Returns:
        - model: The trained PyTorch model
        - data: The dictionary of data splits (tensors + numpy)
        - history: A dictionary containing loss and accuracy metrics per epoch
    """
    # 1. Prepare Data
    data = prepare_data(X, y, random_seed=random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        # Should be provided by model_builder, but safety fallback just in case
        print("Warning: No model provided to train(), creating default.")
        input_size = data["input_size"]
        model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    model.to(device)
    
    # 2. Setup Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epochs": []
    }

    # Optional WandB logging
    run = None
    if use_wandb:
        wandb.login()
        run = wandb.init(project=project, config={"epochs": epochs, "lr": lr})

    print(f"Start training on {device} with seed {random_seed}")

    X_train = data["X_train_tensor"]
    y_train = data["y_train_tensor"]
    X_val = data["X_val_tensor"]
    y_val = data["y_val_tensor"]

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X_train)
        loss = criterion(output, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics calculation
        with torch.no_grad():
            train_preds = (output > 0.5).float()
            train_acc = (train_preds == y_train).float().mean()

            model.eval()
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_preds = (val_output > 0.5).float()
            val_acc = (val_preds == y_val).float().mean()

        # Store history
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc.item())
        history["val_loss"].append(val_loss.item())
        history["val_acc"].append(val_acc.item())

        # WandB logging
        if use_wandb and run:
            run.log({
                "Train Loss": loss.item(), "Train Accuracy": train_acc.item(),
                "Val Loss": val_loss.item(), "Val Accuracy": val_acc.item()
            })
            
        # Update Gradio progress bar if callback provided
        if callback_progress:
            callback_progress((epoch + 1) / epochs, f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    if run:
        run.finish()

    return model, data, history