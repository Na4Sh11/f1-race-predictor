"""
Data Loader for F1 Race Data
Fetches data from Kaggle F1 dataset and Ergast API
"""

import pandas as pd
import requests
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1DataLoader:
    """Load and manage F1 racing data"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ergast_base_url = "http://ergast.com/api/f1"
        
    def load_kaggle_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Load F1 dataset from Kaggle
        Dataset: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
        
        Instructions:
        1. Download dataset from Kaggle
        2. Place CSV files in data/raw/
        3. Run this function
        """
        
        dataframes = {}
        
        # Expected CSV files from Kaggle F1 dataset
        csv_files = [
            'circuits.csv',
            'constructor_results.csv',
            'constructor_standings.csv',
            'constructors.csv',
            'driver_standings.csv',
            'drivers.csv',
            'lap_times.csv',
            'pit_stops.csv',
            'qualifying.csv',
            'races.csv',
            'results.csv',
            'seasons.csv',
            'sprint_results.csv',
            'status.csv'
        ]
        
        logger.info("Loading Kaggle F1 dataset...")
        
        for csv_file in csv_files:
            file_path = self.data_dir / csv_file
            if file_path.exists():
                dataframes[csv_file.replace('.csv', '')] = pd.read_csv(file_path)
                logger.info(f"✅ Loaded {csv_file}")
            else:
                logger.warning(f"⚠️  {csv_file} not found")
        
        return dataframes
    
    def fetch_ergast_data(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Fetch data from Ergast API
        API Docs: http://ergast.com/mrd/
        """
        url = f"{self.ergast_base_url}/{endpoint}.json"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from Ergast API: {e}")
            return {}
    
    def get_season_results(self, year: int) -> pd.DataFrame:
        """Get all race results for a specific season"""
        data = self.fetch_ergast_data(f"{year}/results")
        
        if not data:
            return pd.DataFrame()
        
        races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        
        results = []
        for race in races:
            race_name = race['raceName']
            circuit = race['Circuit']['circuitName']
            date = race['date']
            
            for result in race['Results']:
                results.append({
                    'year': year,
                    'race_name': race_name,
                    'circuit': circuit,
                    'date': date,
                    'position': result.get('position'),
                    'driver': result['Driver']['familyName'],
                    'constructor': result['Constructor']['name'],
                    'points': result.get('points', 0),
                    'status': result['status']
                })
        
        return pd.DataFrame(results)
    
    def merge_datasets(self, kaggle_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple datasets into a comprehensive racing dataset
        """
        logger.info("Merging datasets...")
        
        # Start with results as base
        df = kaggle_data['results'].copy()
        
        # Merge with races info
        df = df.merge(
            kaggle_data['races'][['raceId', 'year', 'round', 'circuitId', 'name', 'date']],
            on='raceId',
            how='left'
        )
        
        # Merge with driver info
        df = df.merge(
            kaggle_data['drivers'][['driverId', 'driverRef', 'number', 'code', 'forename', 'surname', 'dob', 'nationality']],
            on='driverId',
            how='left'
        )
        
        # Merge with constructor info
        df = df.merge(
            kaggle_data['constructors'][['constructorId', 'constructorRef', 'name', 'nationality']],
            on='constructorId',
            how='left',
            suffixes=('', '_constructor')
        )
        
        # Merge with circuit info
        df = df.merge(
            kaggle_data['circuits'][['circuitId', 'circuitRef', 'name', 'location', 'country', 'lat', 'lng']],
            on='circuitId',
            how='left',
            suffixes=('', '_circuit')
        )
        
        # Merge with qualifying data
        if 'qualifying' in kaggle_data:
            qualifying = kaggle_data['qualifying'][['qualifyId', 'raceId', 'driverId', 'constructorId', 'position']]
            qualifying.rename(columns={'position': 'quali_position'}, inplace=True)
            df = df.merge(qualifying, on=['raceId', 'driverId', 'constructorId'], how='left')
        
        logger.info(f"✅ Merged dataset shape: {df.shape}")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data"""
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"✅ Saved processed data to {output_path}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data"""
        file_path = Path("data/processed") / filename
        
        if file_path.exists():
            logger.info(f"Loading processed data from {file_path}")
            return pd.read_parquet(file_path)
        else:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()


def main():
    """Example usage"""
    loader = F1DataLoader()
    
    # Load Kaggle data
    kaggle_data = loader.load_kaggle_dataset()
    
    if kaggle_data:
        # Merge datasets
        merged_df = loader.merge_datasets(kaggle_data)
        
        # Save processed data
        loader.save_processed_data(merged_df, "f1_merged_data.parquet")
        
        print(f"\n📊 Dataset Info:")
        print(f"Shape: {merged_df.shape}")
        print(f"Years covered: {merged_df['year'].min()} - {merged_df['year'].max()}")
        print(f"Total races: {merged_df['raceId'].nunique()}")
        print(f"\nColumns: {list(merged_df.columns)}")
        print(f"\n✅ Data loading complete!")


if __name__ == "__main__":
    main()