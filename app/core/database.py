"""
Database models and connection management
SQLAlchemy with async support for production scalability
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)

# Database base
Base = declarative_base()

# Database models
class SummarizationRequest(Base):
    """Track summarization requests for analytics and caching"""
    __tablename__ = "summarization_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    content_hash = Column(String(64), index=True, nullable=False)
    content_type = Column(String(20), nullable=False)  # text, pdf, youtube
    content_length = Column(Integer, nullable=False)
    
    # Request parameters
    query = Column(Text, nullable=True)
    mode = Column(String(20), nullable=True)
    max_length = Column(Integer, nullable=True)
    min_length = Column(Integer, nullable=True)
    
    # Results
    summary_short = Column(Text, nullable=True)
    summary_medium = Column(Text, nullable=True)
    summary_detailed = Column(Text, nullable=True)
    key_points = Column(JSON, nullable=True)
    confidence_scores = Column(JSON, nullable=True)
    
    # Metadata
    processing_time = Column(Float, nullable=True)
    models_used = Column(JSON, nullable=True)
    cache_hit = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class YouTubeVideo(Base):
    """Track YouTube video processing"""
    __tablename__ = "youtube_videos"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(20), unique=True, index=True, nullable=False)
    title = Column(Text, nullable=True)
    duration = Column(Integer, nullable=True)  # seconds
    
    # Transcript info
    transcript_available = Column(Boolean, default=False)
    transcript_language = Column(String(10), nullable=True)
    transcript_source = Column(String(20), nullable=True)  # youtube, whisper
    
    # Processing status
    processed_at = Column(DateTime, nullable=True)
    processing_error = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserFeedback(Base):
    """Store user feedback for model improvement"""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, nullable=False)  # Reference to SummarizationRequest
    
    # Feedback data
    rating = Column(Integer, nullable=False)  # 1-5 scale
    feedback_text = Column(Text, nullable=True)
    feedback_type = Column(String(20), nullable=False)  # quality, accuracy, relevance
    
    # User context (optional)
    user_mode = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

# Database engine and session
engine = None
async_session_maker = None

async def init_db():
    """Initialize database connection and create tables"""
    global engine, async_session_maker
    
    try:
        # Create async engine
        if settings.DATABASE_URL.startswith("sqlite"):
            # SQLite async URL
            db_url = settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
        else:
            # PostgreSQL async URL
            db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        
        engine = create_async_engine(
            db_url,
            echo=settings.DEBUG,
            future=True
        )
        
        # Create session maker
        async_session_maker = async_sessionmaker(
            engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def get_db_session() -> AsyncSession:
    """Get database session"""
    if not async_session_maker:
        raise RuntimeError("Database not initialized")
    
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

# Database operations
class DatabaseManager:
    """Database operations manager"""
    
    @staticmethod
    async def save_summarization_request(
        content_hash: str,
        content_type: str,
        content_length: int,
        summary_data: Dict[str, Any],
        processing_time: float,
        models_used: list,
        cache_hit: bool = False,
        **kwargs
    ) -> int:
        """Save summarization request to database"""
        try:
            async with async_session_maker() as session:
                request = SummarizationRequest(
                    content_hash=content_hash,
                    content_type=content_type,
                    content_length=content_length,
                    query=kwargs.get('query'),
                    mode=kwargs.get('mode'),
                    max_length=kwargs.get('max_length'),
                    min_length=kwargs.get('min_length'),
                    summary_short=summary_data.get('summary_short'),
                    summary_medium=summary_data.get('summary_medium'),
                    summary_detailed=summary_data.get('summary_detailed'),
                    key_points=summary_data.get('key_points'),
                    confidence_scores=summary_data.get('confidence_scores'),
                    processing_time=processing_time,
                    models_used=models_used,
                    cache_hit=cache_hit
                )
                
                session.add(request)
                await session.commit()
                await session.refresh(request)
                
                return request.id
                
        except Exception as e:
            logger.error(f"Database save error: {e}")
            return -1
    
    @staticmethod
    async def save_youtube_video(
        video_id: str,
        title: Optional[str] = None,
        duration: Optional[int] = None,
        transcript_available: bool = False,
        transcript_language: Optional[str] = None,
        transcript_source: Optional[str] = None
    ) -> int:
        """Save YouTube video information"""
        try:
            async with async_session_maker() as session:
                # Check if video exists
                existing = await session.get(YouTubeVideo, {"video_id": video_id})
                
                if existing:
                    # Update existing
                    existing.title = title or existing.title
                    existing.duration = duration or existing.duration
                    existing.transcript_available = transcript_available
                    existing.transcript_language = transcript_language
                    existing.transcript_source = transcript_source
                    existing.processed_at = datetime.utcnow()
                    video = existing
                else:
                    # Create new
                    video = YouTubeVideo(
                        video_id=video_id,
                        title=title,
                        duration=duration,
                        transcript_available=transcript_available,
                        transcript_language=transcript_language,
                        transcript_source=transcript_source,
                        processed_at=datetime.utcnow()
                    )
                    session.add(video)
                
                await session.commit()
                await session.refresh(video)
                
                return video.id
                
        except Exception as e:
            logger.error(f"YouTube video save error: {e}")
            return -1
    
    @staticmethod
    async def save_user_feedback(
        request_id: int,
        rating: int,
        feedback_text: Optional[str] = None,
        feedback_type: str = "quality",
        user_mode: Optional[str] = None
    ) -> int:
        """Save user feedback"""
        try:
            async with async_session_maker() as session:
                feedback = UserFeedback(
                    request_id=request_id,
                    rating=rating,
                    feedback_text=feedback_text,
                    feedback_type=feedback_type,
                    user_mode=user_mode
                )
                
                session.add(feedback)
                await session.commit()
                await session.refresh(feedback)
                
                return feedback.id
                
        except Exception as e:
            logger.error(f"Feedback save error: {e}")
            return -1