import os
import torch

def save_weights(model, m_name, lr, batch, epoch):
    """
    Saves the model's state_dict (weights and biases) to a .pth file.
    This acts as a 'Save Point' for long training sessions.
    """
    weight_dir = os.path.join("results", m_name, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    
    # Construct filename including epoch and hyperparameters
    path = os.path.join(weight_dir, f"weights_LR{lr}_B{batch}_epoch_{epoch}.pth")
    
    # Save parameters only (state_dict) to minimize file size
    torch.save(model.state_dict(), path)
    print(f"[System] Saved weights to {path}")

def load_weights(model, m_name, lr, batch, epoch, device='cpu'):
    """
    Loads saved parameters into a model instance. 
    Required for resuming training or final testing.
    """
    path = os.path.join("results", m_name, "weights", f"weights_LR{lr}_B{batch}_epoch_{epoch}.pth")
    
    if os.path.exists(path):
        # map_location ensures hardware compatibility (e.g., loading GPU weights on CPU)
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[System] Resumed {m_name} weights from Epoch {epoch}")
        return model
    else:
        print(f"[Error] File not found: {path}")
        return None