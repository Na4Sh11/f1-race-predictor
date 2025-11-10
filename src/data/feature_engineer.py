"""
Feature Engineering for F1 Race Prediction
Creates time-series features, driver/constructor performance metrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1FeatureEngineer:
    """Generate features for F1 race prediction"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['date', 'raceId'])
        
    def create_driver_performance_features(self, window_sizes: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """
        Create rolling performance features for drivers
        - Average position in last N races
        - Points earned in last N races
        - Win rate, podium rate
        """
        logger.info("Creating driver performance features...")
        
        df = self.df.copy()
        
        # Convert position to numeric
        df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
        df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)
        
        features = []
        
        for driver_id in df['driverId'].unique():
            driver_data = df[df['driverId'] == driver_id].sort_values('date').copy()
            
            for window in window_sizes:
                # Rolling average position
                driver_data[f'driver_avg_position_last_{window}'] = (
                    driver_data['position_numeric'].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling average points
                driver_data[f'driver_avg_points_last_{window}'] = (
                    driver_data['points'].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling win rate
                driver_data[f'driver_win_rate_last_{window}'] = (
                    (driver_data['position_numeric'] == 1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                # Rolling podium rate (top 3)
                driver_data[f'driver_podium_rate_last_{window}'] = (
                    (driver_data['position_numeric'] <= 3)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                # Rolling DNF rate
                driver_data[f'driver_dnf_rate_last_{window}'] = (
                    (driver_data['statusId'] != 1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            
            features.append(driver_data)
        
        return pd.concat(features, ignore_index=True)
    
    def create_constructor_performance_features(self, window_sizes: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Create rolling performance features for constructors"""
        logger.info("Creating constructor performance features...")
        
        df = self.df.copy()
        
        features = []
        
        for constructor_id in df['constructorId'].unique():
            constructor_data = df[df['constructorId'] == constructor_id].sort_values('date').copy()
            
            for window in window_sizes:
                # Rolling average position
                constructor_data[f'constructor_avg_position_last_{window}'] = (
                    constructor_data['position_numeric'].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling average points
                constructor_data[f'constructor_avg_points_last_{window}'] = (
                    constructor_data['points'].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling win rate
                constructor_data[f'constructor_win_rate_last_{window}'] = (
                    (constructor_data['position_numeric'] == 1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            
            features.append(constructor_data)
        
        return pd.concat(features, ignore_index=True)
    
    def create_circuit_features(self) -> pd.DataFrame:
        """Create circuit-specific features"""
        logger.info("Creating circuit features...")
        
        df = self.df.copy()
        
        # Circuit type (street, permanent, mixed)
        circuit_types = {
            'Monaco': 'street',
            'Singapore': 'street',
            'Baku': 'street',
            'Montreal': 'street',
            # Add more circuit classifications
        }
        
        df['circuit_type'] = df['name_circuit'].map(circuit_types).fillna('permanent')
        
        # Historical performance at circuit
        for driver_id in df['driverId'].unique():
            for circuit_id in df['circuitId'].unique():
                mask = (df['driverId'] == driver_id) & (df['circuitId'] == circuit_id)
                circuit_driver_data = df[mask].sort_values('date')
                
                # Average position at this circuit
                df.loc[mask, 'driver_circuit_avg_position'] = (
                    circuit_driver_data['position_numeric'].expanding().mean()
                )
                
                # Number of times driven at this circuit
                df.loc[mask, 'driver_circuit_experience'] = range(1, len(circuit_driver_data) + 1)
        
        return df
    
    def create_qualifying_features(self) -> pd.DataFrame:
        """Create qualifying-related features"""
        logger.info("Creating qualifying features...")
        
        df = self.df.copy()
        
        if 'quali_position' in df.columns:
            df['quali_position'] = pd.to_numeric(df['quali_position'], errors='coerce')
            
            # Position gain/loss from qualifying to race
            df['position_change'] = df['quali_position'] - df['position_numeric']
            
            # Front row start (P1 or P2)
            df['front_row_start'] = (df['quali_position'] <= 2).astype(int)
            
            # Top 10 start
            df['q3_start'] = (df['quali_position'] <= 10).astype(int)
        
        return df
    
    def create_temporal_features(self) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        df = self.df.copy()
        
        # Race number in season
        df['race_number'] = df.groupby('year').cumcount() + 1
        
        # Season progress (0 to 1)
        df['season_progress'] = df.groupby('year')['race_number'].transform(
            lambda x: x / x.max()
        )
        
        # Days since season start
        df['days_since_season_start'] = (
            df.groupby('year')['date'].transform(lambda x: (x - x.min()).dt.days)
        )
        
        return df
    
    def create_momentum_features(self) -> pd.DataFrame:
        """Create momentum and trend features"""
        logger.info("Creating momentum features...")
        
        df = self.df.copy()
        
        # Recent 3-race trend
        for driver_id in df['driverId'].unique():
            driver_data = df[df['driverId'] == driver_id].sort_values('date').copy()
            
            # Position trend (improving = negative, declining = positive)
            driver_data['position_trend_3races'] = (
                driver_data['position_numeric'].diff(3)
            )
            
            # Consecutive podiums
            podium = (driver_data['position_numeric'] <= 3).astype(int)
            driver_data['consecutive_podiums'] = (
                podium.groupby((podium != podium.shift()).cumsum()).cumsum()
            )
            
            df.loc[df['driverId'] == driver_id, 'position_trend_3races'] = driver_data['position_trend_3races']
            df.loc[df['driverId'] == driver_id, 'consecutive_podiums'] = driver_data['consecutive_podiums']
        
        return df
    
    def create_championship_standing_features(self) -> pd.DataFrame:
        """Create features based on championship standings"""
        logger.info("Creating championship standing features...")
        
        df = self.df.copy()
        
        # Cumulative points in season
        df['season_points_cumulative'] = (
            df.groupby(['year', 'driverId'])['points'].cumsum()
        )
        
        # Position in championship at time of race
        df['championship_position'] = (
            df.groupby(['year', 'raceId'])['season_points_cumulative']
            .rank(ascending=False, method='min')
        )
        
        # Points gap to leader
        df['points_to_leader'] = (
            df.groupby(['year', 'raceId'])['season_points_cumulative']
            .transform('max') - df['season_points_cumulative']
        )
        
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Run all feature engineering steps"""
        logger.info("🔧 Starting feature engineering pipeline...")
        
        # Ensure position_numeric exists
        self.df['position_numeric'] = pd.to_numeric(self.df['position'], errors='coerce')
        
        # Run all feature engineering methods
        df = self.create_driver_performance_features()
        self.df = df
        
        df = self.create_constructor_performance_features()
        self.df = df
        
        df = self.create_circuit_features()
        self.df = df
        
        df = self.create_qualifying_features()
        self.df = df
        
        df = self.create_temporal_features()
        self.df = df
        
        df = self.create_momentum_features()
        self.df = df
        
        df = self.create_championship_standing_features()
        self.df = df
        
        logger.info(f"✅ Feature engineering complete! Shape: {df.shape}")
        logger.info(f"📊 Total features: {len(df.columns)}")
        
        return df


def main():
    """Example usage"""
    from pathlib import Path
    
    # Load processed data
    df = pd.read_parquet("data/processed/f1_merged_data.parquet")
    
    # Initialize feature engineer
    feature_engineer = F1FeatureEngineer(df)
    
    # Create all features
    df_features = feature_engineer.engineer_all_features()
    
    # Save feature-engineered dataset
    output_path = Path("data/features/f1_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    
    print(f"\n✅ Features saved to {output_path}")
    print(f"📊 Dataset shape: {df_features.shape}")
    print(f"\n🔧 Feature columns:")
    for col in df_features.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()