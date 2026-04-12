import pandas as pd
import matplotlib.pyplot as plt
import os

class ExperimentVisualizer:
    def __init__(self, lr, batch, train_samples=116945):
        self.lr = lr
        self.batch = batch
        self.train_samples = train_samples
        # Professional color palette for distinct model comparison
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_summary(self, model_names):
        """
        Plots Epoch-level Loss, Training Accuracy, and Validation Accuracy side-by-side.
        Useful for comparing overall performance across multiple architectures.
        """
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        steps_per_epoch = self.train_samples // self.batch

        for i, name in enumerate(model_names):
            acc_path = f"results/{name}/metrics/accuracy_LR{self.lr}_B{self.batch}_summary.csv"
            loss_path = f"results/{name}/metrics/loss_LR{self.lr}_B{self.batch}_detailed.csv"
            
            if os.path.exists(acc_path) and os.path.exists(loss_path):
                df_acc = pd.read_csv(acc_path)
                df_loss = pd.read_csv(loss_path)
                epochs = range(1, len(df_acc) + 1)

                # Calculate Average Loss per Epoch from batch-level data
                epoch_losses = [
                    df_loss['batch_loss'].iloc[e*steps_per_epoch : (e+1)*steps_per_epoch].mean()
                    for e in range(len(df_acc))
                ]

                # Plotting data on respective axes
                axes[0].plot(epochs, epoch_losses, label=name, marker='o', color=self.colors[i % len(self.colors)])
                axes[1].plot(epochs, df_acc['train_acc'], label=name, marker='o', color=self.colors[i % len(self.colors)])
                axes[2].plot(epochs, df_acc['val_acc'], label=name, marker='o', color=self.colors[i % len(self.colors)])

        # Set chart aesthetics
        titles = ['Avg Training Loss', 'Training Accuracy (%)', 'Validation Accuracy (%)']
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_overfitting(self, model_names):
        """
        Plots the 'Gap' between Train and Validation accuracy for each model.
        Arranges plots in a single row for direct comparison of generalization.
        """
        num_models = len(model_names)
        # Create subplots in 1 row, with columns equal to the number of models
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
        
        # Ensure axes is iterable even if there is only one model
        if num_models == 1:
            axes = [axes]

        for i, name in enumerate(model_names):
            path = f"results/{name}/metrics/accuracy_LR{self.lr}_B{self.batch}_summary.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                epochs = range(1, len(df) + 1)
                
                # Plot Train vs Val
                axes[i].plot(epochs, df['train_acc'], 'o-', label='Train', color='blue')
                axes[i].plot(epochs, df['val_acc'], 's--', label='Val', color='red')
                
                # Highlight the Overfitting Gap (the area between lines)
                axes[i].fill_between(epochs, df['train_acc'], df['val_acc'], color='gray', alpha=0.1)
                
                axes[i].set_title(f'Overfitting: {name}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Epochs')
                axes[i].set_ylabel('Accuracy (%)')
                axes[i].legend(loc='lower right')
                axes[i].grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.show()