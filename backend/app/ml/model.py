# unchanged core LSTM

import torch
import torch.nn as nn

# This single class can be used for both options and futures,
# as their architecture is the same.
class LSTMSignalModel(nn.Module):
    """A general LSTM model for time-series prediction."""
    def __init__(self, input_size: int, hidden_layer_size: int = 100, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        return self.linear(lstm_out[:, -1, :])
