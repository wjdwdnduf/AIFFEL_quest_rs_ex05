import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    """
    [Baseline] Simple Recurrent Neural Network.
    Vulnerable to Vanishing Gradients in long sentences.
    """
    def __init__(self, vocab_size, word_vector_dim, hidden_dim):
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_vector_dim)
        self.rnn = nn.RNN(word_vector_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        # We only use the last hidden state as the sentence summary
        _, h_n = self.rnn(x)
        x = h_n[-1] 
        return self.sigmoid(self.fc(x))

class LSTMModel(nn.Module):
    """
    [Standard] Long Short-Term Memory Network.
    Uses Gates to preserve information over longer sequences.
    """
    def __init__(self, vocab_size, word_vector_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_vector_dim)
        # LSTM returns (output, (h_n, c_n))
        self.lstm = nn.LSTM(word_vector_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        # h_n is the final hidden state, c_n is the final cell state
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        return self.sigmoid(self.fc(x))