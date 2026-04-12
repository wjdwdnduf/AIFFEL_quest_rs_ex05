import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """
    [Advanced] CNN + LSTM Architecture.
    CNN extracts spatial features, LSTM processes them as a sequence.
    """
    def __init__(self, vocab_size, word_vector_dim, hidden_dim):
        super(HybridModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_vector_dim)
        # CNN layer to reduce sequence complexity
        self.conv = nn.Conv1d(word_vector_dim, word_vector_dim, kernel_size=3, padding=1)
        # LSTM to capture long-term dependencies of the CNN features
        self.lstm = nn.LSTM(word_vector_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x)).permute(0, 2, 1) # Back to [Batch, Seq, Dim]
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        return self.sigmoid(self.fc(x))