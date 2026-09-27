from fastapi import FastAPI
from app.api.routes import router
from app.api.analysis_routes import router as analysis_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Explainable real-time fake news and misinformation detection agent "
        "with evidence retrieval, risk scoring and explainability."
    )
)

app.include_router(router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Fake News Misinformation Detection Agent API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
        "analyze": "/api/v1/analyze"
    }