"""
Complete Training Pipeline for F1 Race Prediction
Supports both LSTM and Transformer models with MLflow tracking
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from pathlib import Path
import yaml
import logging
from typing import Dict, Tuple, List
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1Dataset(Dataset):
    """PyTorch Dataset for F1 race data"""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 5
    ):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Create sequences
        self.X, self.y = self._create_sequences()
        
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time-series prediction"""
        X, y = [], []
        
        for i in range(len(self.features) - self.sequence_length):
            X.append(self.features[i:i + self.sequence_length])
            y.append(self.targets[i + self.sequence_length])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class F1Trainer:
    """Complete training pipeline for F1 models"""
    
    def __init__(
        self,
        model,
        config: Dict,
        device: str = None
    ):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = StandardScaler()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.best_val_loss = float('inf')
        
        # Setup
        self._setup_training()
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        
        # Scheduler
        if self.config.get('scheduler') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        elif self.config.get('scheduler') == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'position_numeric',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test dataloaders"""
        
        logger.info("Preparing data...")
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Fill missing features with median
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Extract features and targets
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data chronologically (important for time series!)
        train_size = int(len(X_scaled) * (1 - test_size - val_size))
        val_size_abs = int(len(X_scaled) * val_size)
        
        X_train = X_scaled[:train_size]
        y_train = y[:train_size]
        
        X_val = X_scaled[train_size:train_size + val_size_abs]
        y_val = y[train_size:train_size + val_size_abs]
        
        X_test = X_scaled[train_size + val_size_abs:]
        y_test = y[train_size + val_size_abs:]
        
        # Create datasets
        train_dataset = F1Dataset(X_train, y_train, self.config['sequence_length'])
        val_dataset = F1Dataset(X_val, y_val, self.config['sequence_length'])
        test_dataset = F1Dataset(X_test, y_test, self.config['sequence_length'])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"✅ Data prepared:")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset)}")
        logger.info(f"  Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('clip_grad'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['clip_grad']
                    )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                all_predictions.extend(output.squeeze().cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        
        return avg_loss, mae
    
    def train(
        self,
        train_loader,
        val_loader,
        experiment_name: str = "f1_race_prediction"
    ):
        """Complete training loop with MLflow tracking"""
        
        logger.info(f"🚀 Starting training on {self.device}")
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)
            mlflow.log_param("device", self.device)
            
            for epoch in range(self.config['epochs']):
                logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
                
                # Train
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss, val_mae = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_maes.append(val_mae)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                logger.info(f"Train Loss: {train_loss:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('models/saved_models/best_model.pt', epoch, val_loss)
                    mlflow.pytorch.log_model(self.model, "best_model")
                    logger.info(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if self.config.get('early_stopping'):
                    if len(self.val_losses) > self.config['early_stopping_patience']:
                        recent_losses = self.val_losses[-self.config['early_stopping_patience']:]
                        if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                            logger.info("Early stopping triggered!")
                            break
            
            # Save final metrics
            final_metrics = {
                'final_train_loss': self.train_losses[-1],
                'final_val_loss': self.val_losses[-1],
                'final_val_mae': self.val_maes[-1],
                'best_val_loss': self.best_val_loss
            }
            mlflow.log_metrics(final_metrics)
            
            logger.info("\n✅ Training complete!")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Save model checkpoint"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'scaler': self.scaler
        }, filepath)


def load_config(config_path: str = "configs/training_config.yaml") -> Dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training script"""
    # Load config
    config = {
        'model_type': 'lstm',  # or 'transformer'
        'sequence_length': 5,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'clip_grad': 1.0,
        'early_stopping': True,
        'early_stopping_patience': 10
    }
    
    # Load data
    df = pd.read_parquet("data/features/f1_features.parquet")
    
    # Select feature columns (exclude IDs, names, targets)
    feature_cols = [col for col in df.columns if col not in [
        'raceId', 'driverId', 'constructorId', 'resultId', 'qualifyId',
        'position', 'position_numeric', 'positionText', 'positionOrder',
        'date', 'name', 'forename', 'surname', 'driverRef', 'constructorRef'
    ] and df[col].dtype in ['float64', 'int64']]
    
    logger.info(f"📊 Using {len(feature_cols)} features")
    
    # Create model
    if config['model_type'] == 'lstm':
        from src.models.lstm_model import F1RaceLSTM
        model = F1RaceLSTM(
            input_dim=len(feature_cols),
            hidden_dim=128,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            use_attention=True
        )
    else:
        from src.models.transformer_model import F1RaceTransformer
        model = F1RaceTransformer(
            input_dim=len(feature_cols),
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dropout=0.3
        )
    
    # Create trainer
    trainer = F1Trainer(model, config)
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        df, feature_cols, target_col='position_numeric'
    )
    
    # Train
    trainer.train(train_loader, val_loader, experiment_name="f1_race_prediction")


if __name__ == "__main__":
    main()