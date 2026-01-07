"""
User Feedback Endpoints
Collect and manage user feedback for model improvement
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from app.api.v1.models import FeedbackRequest, FeedbackResponse
from app.core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/",
    response_model=FeedbackResponse,
    summary="Submit User Feedback",
    description="Submit feedback on summarization quality for model improvement"
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit user feedback on summarization results
    
    - **request_id**: ID of the summarization request being rated
    - **rating**: Rating from 1 (poor) to 5 (excellent)
    - **feedback_text**: Optional detailed feedback
    - **feedback_type**: Type of feedback (quality, accuracy, relevance)
    - **user_mode**: User mode when feedback was given
    
    Feedback is used to:
    - Improve model performance
    - Identify common issues
    - Optimize summarization parameters
    - Train better models
    """
    
    try:
        logger.info(f"Feedback submission: request_id={request.request_id}, rating={request.rating}")
        
        # Save feedback to database
        feedback_id = await DatabaseManager.save_user_feedback(
            request_id=request.request_id,
            rating=request.rating,
            feedback_text=request.feedback_text,
            feedback_type=request.feedback_type,
            user_mode=request.user_mode.value if request.user_mode else None
        )
        
        if feedback_id == -1:
            raise HTTPException(
                status_code=500,
                detail="Failed to save feedback"
            )
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Thank you for your feedback! It helps us improve our summarization quality."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )

@router.get(
    "/stats",
    summary="Feedback Statistics",
    description="Get aggregated feedback statistics (admin only)"
)
async def feedback_stats() -> Dict[str, Any]:
    """
    Get aggregated feedback statistics
    
    Returns:
    - Average ratings by feedback type
    - Rating distribution
    - Common feedback themes
    - Improvement trends
    """
    
    try:
        # This would typically require admin authentication
        # For now, returning mock data structure
        
        return {
            "total_feedback": 0,
            "average_rating": 0.0,
            "rating_distribution": {
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0
            },
            "feedback_by_type": {
                "quality": {"count": 0, "avg_rating": 0.0},
                "accuracy": {"count": 0, "avg_rating": 0.0},
                "relevance": {"count": 0, "avg_rating": 0.0}
            },
            "feedback_by_mode": {
                "student": {"count": 0, "avg_rating": 0.0},
                "researcher": {"count": 0, "avg_rating": 0.0},
                "business": {"count": 0, "avg_rating": 0.0},
                "beginner": {"count": 0, "avg_rating": 0.0},
                "expert": {"count": 0, "avg_rating": 0.0}
            }
        }
        
    except Exception as e:
        logger.error(f"Feedback stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feedback stats: {str(e)}"
        )