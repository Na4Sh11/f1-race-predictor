import torch
import pandas as pd
import numpy as np
from src.models.lstm_model import F1RaceLSTM

print("🔍 Testing F1 Model Performance\n")

# Load model
checkpoint = torch.load("models/saved_models/best_model.pt")
print(f"✅ Model from Epoch: {checkpoint['epoch']}")
print(f"📊 Val Loss: {checkpoint['loss']:.4f}\n")

# Load data
df = pd.read_parquet("data/features/f1_features.parquet")
recent = df[df['year'] >= 2020].dropna(subset=['position_numeric']).head(100)

# Get features
feature_cols = [col for col in df.columns if col not in [
    'raceId', 'driverId', 'constructorId', 'resultId', 'qualifyId',
    'position', 'position_numeric', 'positionText', 'positionOrder',
    'date', 'name', 'forename', 'surname', 'driverRef', 'constructorRef'
] and df[col].dtype in ['float64', 'int64']]

X = recent[feature_cols].fillna(0).values
y = recent['position_numeric'].values

# Load model
model = F1RaceLSTM(input_dim=len(feature_cols), hidden_dim=128, num_layers=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predictions
predictions = []
print("Making predictions...")
with torch.no_grad():
    for i in range(len(X)):
        seq = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(0).repeat(1, 5, 1)
        pred = model(seq).item()
        predictions.append(pred)

predictions = np.array(predictions)
mae = np.mean(np.abs(predictions - y))

print(f"\n{'='*50}")
print(f"📈 MODEL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Mean Absolute Error: {mae:.2f} positions")
print(f"\nSample Predictions:")
for i in range(5):
    print(f"  Actual: P{int(y[i]):2d} → Predicted: P{predictions[i]:.1f}")
print(f"{'='*50}\n")

if mae < 3.0:
    print("✅ Great! Model performance is GOOD")
elif mae < 3.5:
    print("⚠️  Acceptable model performance")
else:
    print("❌ Model needs improvement")
