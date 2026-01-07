"""
Health Check and Monitoring Endpoints
System status, model health, and performance metrics
"""

import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from app.api.v1.models import HealthResponse, CacheStatsResponse, ModelStatsResponse
from app.core.models import ModelManager
from app.core.cache import CacheManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize managers
model_manager = ModelManager()
cache_manager = CacheManager()

@router.get(
    "/",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Comprehensive system health check including models and cache status"
)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check endpoint
    
    Returns:
    - Overall system status
    - Model availability and status
    - Cache connectivity and stats
    - System timestamp and version
    """
    
    try:
        # Check model status
        model_status = await model_manager.health_check()
        
        # Check cache status
        cache_status = await cache_manager.health_check()
        
        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            models=model_status,
            cache=cache_status,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

@router.get(
    "/models",
    response_model=ModelStatsResponse,
    summary="Model Status",
    description="Detailed information about loaded AI models"
)
async def model_status() -> ModelStatsResponse:
    """
    Get detailed model status and statistics
    
    Returns information about:
    - Loaded models and their status
    - Model types and capabilities
    - Memory usage and performance
    - Device information (CPU/GPU)
    """
    
    try:
        model_info = await model_manager.health_check()
        
        return ModelStatsResponse(
            total_models=model_info["total_models"],
            loaded_models=model_info["loaded_models"],
            device=model_info["device"],
            models=model_info["models"]
        )
        
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model status unavailable: {str(e)}"
        )

@router.get(
    "/cache",
    response_model=CacheStatsResponse,
    summary="Cache Statistics",
    description="Cache performance metrics and statistics"
)
async def cache_stats() -> CacheStatsResponse:
    """
    Get cache performance statistics
    
    Returns:
    - Cache connectivity status
    - Memory usage information
    - Hit/miss ratios
    - Key counts by type
    """
    
    try:
        cache_info = await cache_manager.get_cache_stats()
        
        return CacheStatsResponse(
            status=cache_info["status"],
            memory_usage=cache_info.get("memory_usage"),
            total_keys=cache_info.get("total_keys", 0),
            summary_keys=cache_info.get("summary_keys", 0),
            youtube_keys=cache_info.get("youtube_keys", 0),
            hits=cache_info.get("hits", 0),
            misses=cache_info.get("misses", 0)
        )
        
    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache stats unavailable: {str(e)}"
        )

@router.get(
    "/ping",
    summary="Simple Ping",
    description="Simple ping endpoint for basic connectivity testing"
)
async def ping() -> Dict[str, Any]:
    """
    Simple ping endpoint for load balancers and monitoring
    
    Returns basic status and timestamp
    """
    
    return {
        "status": "ok",
        "timestamp": time.time(),
        "message": "pong"
    }

@router.get(
    "/ready",
    summary="Readiness Check",
    description="Check if service is ready to handle requests"
)
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check for Kubernetes and container orchestration
    
    Verifies that:
    - At least one model is loaded
    - Cache is accessible (if configured)
    - System is ready to process requests
    """
    
    try:
        # Check if models are loaded
        model_status = await model_manager.health_check()
        
        if model_status["loaded_models"] == 0:
            raise HTTPException(
                status_code=503,
                detail="No models loaded - service not ready"
            )
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "loaded_models": model_status["loaded_models"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )

@router.get(
    "/live",
    summary="Liveness Check",
    description="Check if service is alive and responding"
)
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check for Kubernetes and container orchestration
    
    Simple check that the service is alive and can respond to requests
    """
    
    return {
        "status": "alive",
        "timestamp": time.time()
    }