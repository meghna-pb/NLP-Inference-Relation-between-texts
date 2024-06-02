import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        """ A simple LSTM (Long Short-Term Memory) network for sequence modeling.

        Args:
            input_dim (int): The size of input vocabulary.
            hidden_dim (int): The size of the hidden state, also the output dimension of the embedding layer.
            output_dim (int): The size of the output layer.
            n_layers (int): The number of stacked LSTM layers.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing sequences of indices, shaped (batch_size, sequence_length).

        Returns:
            out (torch.Vector): Output prediction of the mdoel, shaped (batch_size, output_dim).
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Select the last output
        out = self.fc(lstm_out)
        return out