import pytest
import torch
from src.models.lstm_model import F1RaceLSTM

def test_lstm_forward_pass():
    model = F1RaceLSTM(input_dim=50, hidden_dim=64, num_layers=2)
    x = torch.randn(4, 5, 50)  # batch=4, seq=5, features=50
    output = model(x)
    assert output.shape == (4, 1)
