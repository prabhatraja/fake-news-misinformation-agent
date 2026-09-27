from fastapi import APIRouter
from app.core.config import settings
from app.schemas.prediction_schema import PredictionRequest, PredictionResponse
from app.services.agent_service import run_misinformation_agent

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return run_misinformation_agent(
        text=request.text,
        input_type=request.input_type,
        inference_mode=request.inference_mode,
    )
