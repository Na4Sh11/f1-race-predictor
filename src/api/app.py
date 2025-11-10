"""
FastAPI Application for F1 Race Prediction
Serves predictions via REST API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="F1 Race Position Predictor API",
    description="Predict Formula 1 race finishing positions using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for race prediction"""
    driver_id: int = Field(..., description="Driver ID")
    constructor_id: int = Field(..., description="Constructor/Team ID")
    circuit_id: int = Field(..., description="Circuit ID")
    quali_position: Optional[int] = Field(None, description="Qualifying position")
    year: int = Field(..., description="Season year")
    round: int = Field(..., description="Race round number")
    
    class Config:
        schema_extra = {
            "example": {
                "driver_id": 1,
                "constructor_id": 6,
                "circuit_id": 1,
                "quali_position": 3,
                "year": 2024,
                "round": 5
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_position: float = Field(..., description="Predicted finishing position")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    model_used: str = Field(..., description="Model used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ModelInfo(BaseModel):
    """Model information"""
    model_type: str
    version: str
    trained_on: str
    total_parameters: int
    input_features: int
    status: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool


# Global model storage
class ModelManager:
    """Manage loaded models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_path: str, model_type: str):
        """Load a trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            
            # Load model based on type
            if model_type == 'lstm':
                from src.models.lstm_model import F1RaceLSTM
                config = checkpoint.get('config', {})
                
                # Get input_dim from checkpoint if available
                state_dict = checkpoint['model_state_dict']
                # Extract input_dim from the first weight matrix
                input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
                
                model = F1RaceLSTM(
                    input_dim=input_dim,  # ← Use actual dimension from checkpoint
                    hidden_dim=config.get('hidden_dim', 128),
                    num_layers=config.get('num_layers', 3),
                    dropout=config.get('dropout', 0.3),
                    bidirectional=True,
                    use_attention=True
                )
            elif model_type == 'transformer':
                from src.models.transformer_model import F1RaceTransformer
                config = checkpoint['config']
                model = F1RaceTransformer(
                    input_dim=config.get('input_dim', 50),
                    d_model=config.get('d_model', 128),
                    nhead=config.get('nhead', 8),
                    num_encoder_layers=config.get('num_encoder_layers', 4),
                    dropout=config.get('dropout', 0.3)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models[model_type] = model
            self.scalers[model_type] = checkpoint.get('scaler')
            
            logger.info(f"✅ {model_type.upper()} model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading {model_type} model: {e}")
            return False
    
    def predict(
        self,
        features: np.ndarray,
        model_type: str = 'transformer'
    ) -> tuple:
        """Make prediction using specified model"""
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")
        
        model = self.models[model_type]
        scaler = self.scalers.get(model_type)
        
        # Preprocess features
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Convert to tensor
        features_tensor = torch.tensor(
            features_scaled,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = model(features_tensor)
            predicted_position = prediction.item()
        
        # Calculate confidence (simplified)
        confidence = 1.0 / (1.0 + abs(predicted_position - round(predicted_position)))
        
        return predicted_position, confidence


# Initialize model manager
model_manager = ModelManager()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting F1 Race Predictor API...")
    
    # Try to load models
    model_dir = Path("models/saved_models")
    
    if (model_dir / "best_lstm_model.pt").exists():
        model_manager.load_model(str(model_dir / "best_lstm_model.pt"), "lstm")
    
    if (model_dir / "best_transformer_model.pt").exists():
        model_manager.load_model(str(model_dir / "best_transformer_model.pt"), "transformer")
    
    if not model_manager.models:
        logger.warning("⚠️  No models loaded. Please train models first.")
    else:
        logger.info(f"✅ Loaded models: {list(model_manager.models.keys())}")


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "F1 Race Position Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=len(model_manager.models) > 0
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_race_position(request: PredictionRequest):
    """
    Predict race finishing position
    
    This endpoint takes driver, constructor, circuit information and returns
    a predicted finishing position using the trained model.
    """
    
    start_time = datetime.now()
    
    try:
        # TODO: Implement proper feature extraction from request
        # For now, using dummy features
        # In production, you would:
        # 1. Look up historical data for this driver/circuit
        # 2. Calculate rolling statistics
        # 3. Get qualifying position
        # 4. Generate all engineered features
        
        # Dummy features for demonstration
        features = np.random.randn(1, 50)  # Replace with actual feature extraction
        
        # Make prediction
        model_type = 'transformer' if 'transformer' in model_manager.models else 'lstm'
        predicted_position, confidence = model_manager.predict(features, model_type)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            predicted_position=round(predicted_position, 2),
            confidence=round(confidence, 3),
            model_used=model_type,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Batch prediction endpoint
    Predict positions for multiple drivers in a race
    """
    
    predictions = []
    
    for request in requests:
        try:
            # Make individual prediction
            features = np.random.randn(1, 50)  # Replace with actual features
            model_type = 'transformer' if 'transformer' in model_manager.models else 'lstm'
            predicted_position, confidence = model_manager.predict(features, model_type)
            
            predictions.append({
                "driver_id": request.driver_id,
                "predicted_position": round(predicted_position, 2),
                "confidence": round(confidence, 3)
            })
        except Exception as e:
            logger.error(f"Error predicting for driver {request.driver_id}: {e}")
            predictions.append({
                "driver_id": request.driver_id,
                "error": str(e)
            })
    
    return {"predictions": predictions}


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about loaded models"""
    
    if not model_manager.models:
        raise HTTPException(status_code=404, detail="No models loaded")
    
    # Get first available model info
    model_type = list(model_manager.models.keys())[0]
    model = model_manager.models[model_type]
    
    total_params = sum(p.numel() for p in model.parameters())
    
    return ModelInfo(
        model_type=model_type,
        version="1.0.0",
        trained_on=datetime.now().strftime("%Y-%m-%d"),
        total_parameters=total_params,
        input_features=model.input_dim,
        status="ready"
    )


@app.get("/model/list", tags=["Model"])
async def list_models():
    """List all loaded models"""
    return {
        "available_models": list(model_manager.models.keys()),
        "total_models": len(model_manager.models)
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not found",
        "message": "The requested resource was not found"
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)