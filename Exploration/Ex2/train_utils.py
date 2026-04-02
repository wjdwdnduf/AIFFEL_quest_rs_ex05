import torch
import time

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=1):
    """
    Performs full training and validation cycles, tracking performance metrics.
    
    Returns:
        train_losses: List of individual batch losses (Fine-grained optimization history)
        train_accuracies: List of average training accuracy per epoch (Learning capacity)
        val_accuracies: List of average validation accuracy per epoch (Generalization ability)
    """
    train_losses = []      
    train_accuracies = []  
    val_accuracies = []    

    for epoch in range(epochs):
        # --- PHASE 1: TRAINING ---
        model.train()  # Set the model to training mode (enables Dropout, Batch Norm updates)
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            # Move data to the active device (GPU/MPS/CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear previous gradients to prevent accumulation
            optimizer.zero_grad()
            
            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            
            # Calculate the loss (difference between predictions and ground truth)
            loss = criterion(outputs, labels)
            
            # Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Optimization step: Update model weights based on computed gradients
            optimizer.step()

            # Record batch-level loss for high-resolution visualization
            running_loss += loss.item()
            train_losses.append(loss.item())

            # Calculate training accuracy for the current batch
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Calculate and store average Training Accuracy for the epoch
        train_acc = 100 * train_correct / train_total
        train_accuracies.append(train_acc)

        # --- PHASE 2: VALIDATION ---
        model.eval()  # Set the model to evaluation mode (disables Dropout, freezes Batch Norm)
        val_correct = 0
        val_total = 0
        
        # Disable gradient calculation to save memory and compute power during inference
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Determine the class with the highest probability
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate and store average Validation Accuracy for the epoch
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)

        # Real-time progress monitoring
        print(f"Epoch [{epoch+1}/{epochs}] Summary:")
        print(f" -> Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    return train_losses, train_accuracies, val_accuracies