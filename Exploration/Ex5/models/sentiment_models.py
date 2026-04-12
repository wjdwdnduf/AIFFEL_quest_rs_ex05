import torch
import torch.nn as nn

class DropoutHybridNet(nn.Module):
    """
    Hybrid architecture combining CNN for local feature extraction 
    and Bidirectional LSTM for global context, with Dropout for regularization.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # CNN Layer: Extracts local n-gram features
        self.conv = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        
        # LSTM Layer: Captures long-term dependencies
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, bidirectional=True)
        
        # Dropout: Randomly zeroes 50% of activations to prevent overfitting
        self.dropout = nn.Dropout(0.5) 
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x).transpose(1, 2) # [batch, emb, seq_len]
        
        conved = nn.functional.relu(self.conv(embedded)).transpose(1, 2)
        
        _, (hidden, _) = self.lstm(conved)
        # Concatenate forward and backward hidden states
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Apply dropout before the final classification
        out = self.dropout(cat) 
        return self.sigmoid(self.fc(out))

# Add your Transformer class here if it's already defined