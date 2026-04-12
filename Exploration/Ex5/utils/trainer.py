import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one full training pass over the dataset (Forward -> Backward -> Optimize).
    """
    model.train() # Enable training mode (activates Dropout/Batchnorm)
    batch_losses = [] 
    correct, total = 0, 0
    
    for inputs, labels in loader:
        # Move data to target hardware (GPU/MPS/CPU)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. Clear previous gradients (Standard PyTorch procedure)
        optimizer.zero_grad()
        
        # 2. Forward pass: Generate predictions
        # squeeze() converts [batch, 1] to [batch] to match labels
        outputs = model(inputs).squeeze() 
        
        # 3. Calculate Loss (Distance between pred and ground truth)
        loss = criterion(outputs, labels)
        
        # 4. Backward pass: Calculate partial derivatives (gradients)
        loss.backward()
        
        # 5. Optimization step: Update weights based on gradients
        optimizer.step()
        
        # Track metrics
        batch_losses.append(loss.item()) # Store loss as a float
        
        # Binary Accuracy Calculation: Threshold at 0.5
        predicted = (outputs > 0.5).float() # 1 if > 0.5, else 0
        total += labels.size(0)            # Running count of total samples
        correct += (predicted == labels).sum().item() # Running count of hits
        
    epoch_acc = 100. * correct / total
    return batch_losses, epoch_acc

def validate(model, loader, criterion, device):
    """
    Evaluates the model on unseen data to monitor generalization.
    Gradients are NOT calculated here to save computational resources.
    """
    model.eval() # Enable evaluation mode (disables Dropout/Batchnorm)
    correct, total = 0, 0
    
    # Disable autograd engine to speed up inference and save memory
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            # Predict labels based on probability
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100. * correct / total


from gensim.models import KeyedVectors, Word2Vec
import torch

def apply_pretrained_embeddings(model, word_to_index, w2v_path, w2v_type):
    """
    Load pre-trained vectors and initialize the model embedding layer.
    Handles both Google binary format and Gensim model format.
    """
    print(f"Loading {w2v_type} vectors from: {w2v_path}")
    
    try:
        if w2v_type == "google":
            # Load Google News binary format
            word2vec = KeyedVectors.load_word2vec_format(w2v_path, binary=True, limit=1000000)
        elif w2v_type == "korean":
            # Load Korean Gensim model format
            ko_model = Word2Vec.load(w2v_path)
            word2vec = ko_model.wv
        else:
            raise ValueError(f"Unknown w2v_type: {w2v_type}")

        embed_size = model.embedding.weight.shape[1]
        embedding_matrix = torch.zeros((len(word_to_index), embed_size))
        
        match_count = 0
        for word, i in word_to_index.items():
            if word in word2vec:
                embedding_matrix[i] = torch.from_numpy(word2vec[word].copy())
                match_count += 1
                
        model.embedding.weight.data.copy_(embedding_matrix)
        coverage = (match_count / len(word_to_index)) * 100
        print(f"Initialization Complete: {match_count} words matched ({coverage:.2f}% coverage).")
        
    except Exception as e:
        print(f"Failed to load embeddings: {e}")