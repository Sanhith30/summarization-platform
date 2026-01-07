"""
API v1 Router
Main router that includes all endpoint modules
"""

from fastapi import APIRouter
from app.api.v1.endpoints import summarization, health, feedback, admin

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    summarization.router,
    prefix="/summarize",
    tags=["summarization"]
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

api_router.include_router(
    feedback.router,
    prefix="/feedback",
    tags=["feedback"]
)

api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"]
)