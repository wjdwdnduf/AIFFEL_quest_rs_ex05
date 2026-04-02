import torch
import torch.nn as nn
import torch.optim as optim

'''
This module provides the training and evaluation engine for ResNet experiments.
It handles the training loop, loss tracking, and accuracy calculation.
'''

def train_and_validate(model, train_loader, test_loader, device, epochs=10):
    # Move the model to the specified device (GPU/MPS/CPU)
    model.to(device)
    
    # Define the loss function for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Standard SGD optimizer: lr=0.1 is used in the original ResNet paper for CIFAR-10
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Corrected Scheduler: MultiStepLR reduces LR at specific milestones (e.g., at epoch 10 and 15)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    
    # Dictionary to store performance metrics for visualization and Ablation Study
    results = {'loss_history': [], 'val_acc': 0.0}

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train() # Set the model to training mode (enables BatchNorm/Dropout)
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Move batch data to the active device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Optimization step
            optimizer.zero_grad()               # Clear gradients from the previous step
            outputs = model(inputs)            # Forward pass: compute predicted outputs
            loss = criterion(outputs, labels)   # Calculate the loss
            loss.backward()                    # Backward pass: compute gradient of the loss
            optimizer.step()                   # Update model parameters
            
            running_loss += loss.item()
            
        # Calculate and store average loss for Task 2 (Loss Monitoring)
        avg_loss = running_loss / len(train_loader)
        results['loss_history'].append(avg_loss)
        
        # Step the scheduler to update the learning rate
        scheduler.step()
        
        # Print progress for every epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - LR: {current_lr:.4f}")

    # --- Validation Phase for Task 3 (Ablation Study) ---
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    
    # Disable gradient calculation to save memory and speed up inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get the index of the max logit (predicted class)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() # Compare with ground truth
            
    # Calculate final validation accuracy percentage
    results['val_acc'] = 100. * correct / total
    print(f"==> Validation Accuracy: {results['val_acc']:.2f}%")
    
    return results