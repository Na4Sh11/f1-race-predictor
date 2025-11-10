import pytest
from src.data.data_loader import F1DataLoader

def test_data_loader_initialization():
    loader = F1DataLoader()
    assert loader is not None
    assert loader.data_dir.exists()
