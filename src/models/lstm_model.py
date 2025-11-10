"""
LSTM Model for F1 Race Position Prediction
Multi-layer LSTM with attention mechanism
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch, seq_len, 1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: (batch, hidden_dim)
        
        return context_vector, attention_weights


class F1RaceLSTM(nn.Module):
    """
    LSTM-based model for predicting F1 race finishing positions
    
    Architecture:
    - Multi-layer bidirectional LSTM
    - Attention mechanism
    - Dropout for regularization
    - Dense layers for prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super(F1RaceLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        if use_attention:
            self.attention = AttentionLayer(lstm_output_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Predict position (1-20)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            predictions: Predicted positions (batch_size, 1)
        """
        # LSTM layers
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_dim * num_directions)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_out)
        else:
            # Use last hidden state
            if self.bidirectional:
                context_vector = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                context_vector = hidden[-1]
        
        # Fully connected layers
        x = self.fc1(context_vector)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer - predicting position (1-20)
        predictions = self.fc3(x)
        
        return predictions


class LSTMTrainer:
    """Training utilities for LSTM model"""
    
    def __init__(
        self,
        model: F1RaceLSTM,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        clip_grad: float = 1.0
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output.squeeze(), target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader, criterion) -> Tuple[float, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                all_predictions.extend(output.squeeze().cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate MAE
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        
        return avg_loss, mae
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer, loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)
        logger.info(f"✅ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, optimizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"✅ Checkpoint loaded from {filepath}")
        return checkpoint['epoch'], checkpoint['loss']


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    sequence_length: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training
    
    Args:
        data: Feature array (n_samples, n_features)
        targets: Target array (n_samples,)
        sequence_length: Number of previous races to consider
    
    Returns:
        X: Sequences (n_samples, sequence_length, n_features)
        y: Targets (n_samples,)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    
    return np.array(X), np.array(y)


def main():
    """Example usage"""
    # Model configuration
    config = {
        'input_dim': 50,  # Number of features
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'bidirectional': True,
        'use_attention': True
    }
    
    # Create model
    model = F1RaceLSTM(**config)
    
    print("🏎️ F1 Race LSTM Model")
    print(f"{'='*50}")
    print(f"Input dimension: {config['input_dim']}")
    print(f"Hidden dimension: {config['hidden_dim']}")
    print(f"Number of layers: {config['num_layers']}")
    print(f"Bidirectional: {config['bidirectional']}")
    print(f"Attention: {config['use_attention']}")
    print(f"{'='*50}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 32
    seq_length = 5
    dummy_input = torch.randn(batch_size, seq_length, config['input_dim'])
    
    output = model(dummy_input)
    print(f"\n✅ Forward pass successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()