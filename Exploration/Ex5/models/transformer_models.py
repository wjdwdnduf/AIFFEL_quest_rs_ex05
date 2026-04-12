import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """
    [Modern] Simple Transformer Encoder.
    Uses Attention mechanisms to weigh the importance of different words.
    """
    def __init__(self, vocab_size, word_vector_dim, nhead=2, num_layers=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_vector_dim)
        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=word_vector_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(word_vector_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling over sequence
        return self.sigmoid(self.fc(x))
    


class RegularizedTransformerModel(nn.Module):
    """
    Transformer Encoder with enhanced regularization.
    Uses higher Dropout and Global Average Pooling.
    """
    def __init__(self, vocab_size, word_vector_dim, nhead=8, num_layers=2, dropout=0.5):
        super(RegularizedTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_vector_dim)
        
        # Increase dropout within the attention and feed-forward layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=word_vector_dim, 
            nhead=nhead, 
            batch_first=True,
            dropout=dropout 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Additional Dropout before final classification
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(word_vector_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) 
        x = self.dropout_layer(x)
        return self.sigmoid(self.fc(x))