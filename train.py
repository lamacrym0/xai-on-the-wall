import wandb
from config import NUM_EPOCHS, LEARNING_RATE

wandb.login()

# Project that the run is recorded to
project = "xai-on-the-wall"

# Dictionary with hyperparameters
config = {
    'epochs' : NUM_EPOCHS,
    'lr' : LEARNING_RATE
}

with wandb.init(project=project, config=config) as run:
    print(f"lr: {config['lr']}")
    
    # Training
    for epoch in range(2, config['epochs']):
        acc = 1 
        loss = 1 
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})