import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    """
    [Fast] 1D Convolutional Neural Network.
    Captures local patterns (n-grams) using filters.
    """
    def __init__(self, vocab_size, word_vector_dim, num_filters=128, kernel_size=3):
        super(CNN1DModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_vector_dim)
        self.conv = nn.Conv1d(word_vector_dim, num_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        self.fc = nn.Linear(num_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1) # [Batch, Dim, Seq]
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)           # [Batch, Filters]
        return self.sigmoid(self.fc(x))

class GlobalMaxPoolModel(nn.Module):
    """
    [Lightweight] Feed-Forward Network using only Global Max Pooling.
    Captures the most prominent feature across the entire sequence.
    """
    def __init__(self, vocab_size, word_vector_dim, hidden_dim=8):
        super(GlobalMaxPoolModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_vector_dim)
        # AdaptiveMaxPool1d(1) is equivalent to GlobalMaxPooling1D
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(word_vector_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Embedding pass: [Batch, Seq, Dim]
        x = self.embedding(x)
        # 2. Permute for Pooling: [Batch, Dim, Seq]
        x = x.permute(0, 2, 1)
        # 3. Pooling: Extracts the max value for each dimension across the sequence
        x = self.global_max_pooling(x)
        # 4. Flatten: [Batch, Dim]
        x = x.view(x.size(0), -1)
        # 5. Fully Connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)