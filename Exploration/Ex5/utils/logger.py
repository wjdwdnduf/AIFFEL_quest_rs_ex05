import os
import pandas as pd

def update_results_refined(m_name, lr, batch, t_loss_list, t_acc, v_acc):
    """
    Saves batch losses and epoch accuracies into CSV files for tracking.
    Uses 'append' mode to support sequential training (resuming from epochs).
    """
    # Create directory: results/[model_name]/metrics
    metric_dir = os.path.join("results", m_name, "metrics")
    os.makedirs(metric_dir, exist_ok=True) # Create folder if it doesn't exist
    
    # Define file paths with hyperparameter metadata in the name
    loss_file = os.path.join(metric_dir, f"loss_LR{lr}_B{batch}_detailed.csv")
    acc_file = os.path.join(metric_dir, f"accuracy_LR{lr}_B{batch}_summary.csv")

    # --- PART A: BATCH-LEVEL LOSS ---
    # Stores loss for every single step; vital for analyzing gradient stability
    new_loss_data = pd.DataFrame({"batch_loss": t_loss_list})
    # 'mode=a' appends to existing CSV; header=False if file already exists
    new_loss_data.to_csv(loss_file, mode='a', header=not os.path.exists(loss_file), index=False)

    # --- PART B: EPOCH-LEVEL ACCURACY ---
    # Stores high-level summary for each completed training pass
    new_acc_data = pd.DataFrame({
        "train_acc": [t_acc], # Current epoch training accuracy
        "val_acc": [v_acc]    # Current epoch validation accuracy
    })
    new_acc_data.to_csv(acc_file, mode='a', header=not os.path.exists(acc_file), index=False)

def load_refined_metric(m_name, metric_type, lr=0.001, batch=64):
    """
    Retrieves logged data from CSV files to prepare for visualization/plotting.
    """
    metric_dir = os.path.join("results", m_name, "metrics")
    fname = f"loss_LR{lr}_B{batch}_detailed.csv" if metric_type == 'loss' else f"accuracy_LR{lr}_B{batch}_summary.csv"
    fpath = os.path.join(metric_dir, fname)
    
    return pd.read_csv(fpath) if os.path.exists(fpath) else pd.DataFrame()