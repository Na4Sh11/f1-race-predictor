"""
Transformer Model for F1 Race Position Prediction
Uses multi-head attention for sequence modeling
"""

import torch
import torch.nn as nn
import math
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class F1RaceTransformer(nn.Module):
    """
    Transformer-based model for F1 race prediction
    
    Architecture:
    - Input embedding with positional encoding
    - Multi-head self-attention layers
    - Feed-forward networks
    - Position prediction head
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_seq_length: int = 50
    ):
        super(F1RaceTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_length,
            dropout=dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better training
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Predict position
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            predictions: Predicted positions (batch_size, 1)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # x shape: (batch, seq_len, d_model)
        
        # Global average pooling over sequence dimension
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Feed-forward layers
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output prediction
        predictions = self.fc3(x)
        
        return predictions


class TransformerWithCrossAttention(nn.Module):
    """
    Advanced Transformer with cross-attention between driver and race features
    Useful when you have separate feature streams
    """
    
    def __init__(
        self,
        driver_feature_dim: int,
        race_feature_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3
    ):
        super(TransformerWithCrossAttention, self).__init__()
        
        # Separate projections for driver and race features
        self.driver_projection = nn.Linear(driver_feature_dim, d_model)
        self.race_projection = nn.Linear(race_feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Cross-attention transformer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, driver_features: torch.Tensor, race_features: torch.Tensor):
        """
        Forward pass with cross-attention
        
        Args:
            driver_features: (batch, seq_len, driver_feature_dim)
            race_features: (batch, seq_len, race_feature_dim)
        """
        # Project features
        driver_emb = self.driver_projection(driver_features)
        race_emb = self.race_projection(race_features)
        
        # Add positional encoding
        driver_emb = self.pos_encoding(driver_emb)
        race_emb = self.pos_encoding(race_emb)
        
        # Cross-attention: driver attends to race context
        attended_features, _ = self.cross_attention(
            query=driver_emb,
            key=race_emb,
            value=race_emb
        )
        
        # Encode combined features
        encoded = self.encoder(attended_features)
        
        # Pool sequence
        driver_pooled = driver_emb.mean(dim=1)
        encoded_pooled = encoded.mean(dim=1)
        
        # Concatenate and predict
        combined = torch.cat([driver_pooled, encoded_pooled], dim=1)
        predictions = self.output_head(combined)
        
        return predictions


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def main():
    """Example usage"""
    # Model configuration
    config = {
        'input_dim': 50,
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.3,
        'max_seq_length': 20
    }
    
    # Create model
    model = F1RaceTransformer(**config)
    
    print("🏎️ F1 Race Transformer Model")
    print(f"{'='*60}")
    print(f"Input dimension: {config['input_dim']}")
    print(f"Model dimension: {config['d_model']}")
    print(f"Number of heads: {config['nhead']}")
    print(f"Number of encoder layers: {config['num_encoder_layers']}")
    print(f"Feedforward dimension: {config['dim_feedforward']}")
    print(f"{'='*60}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 32
    seq_length = 10
    dummy_input = torch.randn(batch_size, seq_length, config['input_dim'])
    
    output = model(dummy_input)
    print(f"\n✅ Forward pass successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compare with LSTM
    print(f"\n🔍 Comparison with LSTM:")
    print(f"Transformers are better at:")
    print(f"  ✓ Capturing long-range dependencies")
    print(f"  ✓ Parallel processing (faster training)")
    print(f"  ✓ Attention visualization")
    print(f"LSTMs are better at:")
    print(f"  ✓ Sequential dependencies")
    print(f"  ✓ Smaller datasets")
    print(f"  ✓ Lower memory usage")


if __name__ == "__main__":
    main()