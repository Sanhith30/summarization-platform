"""
Application configuration management
Environment-based settings with validation
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = Field(default="Summarization Platform")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    
    # API Configuration
    MAX_CONTENT_LENGTH: int = Field(default=1000000)  # 1MB
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    REQUEST_TIMEOUT: int = Field(default=300)  # 5 minutes
    
    # Model Configuration
    MODEL_CACHE_DIR: str = Field(default="./models")
    DEFAULT_MODEL: str = Field(default="facebook/bart-large-cnn")
    ENABLE_GPU: bool = Field(default=True)  # Auto-detect GPU
    MAX_MODEL_MEMORY: str = Field(default="4GB")
    
    # Available models for ensemble (using only BART for speed)
    SUMMARIZATION_MODELS: List[str] = Field(default=[
        "facebook/bart-large-cnn"
    ])
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./app.db")
    
    # Redis Cache
    REDIS_URL: str = Field(default="redis://localhost:6379")
    CACHE_TTL: int = Field(default=3600)  # 1 hour
    
    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    
    # YouTube Processing
    YOUTUBE_MAX_DURATION: int = Field(default=7200)  # 2 hours
    WHISPER_MODEL: str = Field(default="base")
    
    # Summarization Settings
    MAX_CHUNK_SIZE: int = Field(default=1024)
    MIN_CHUNK_SIZE: int = Field(default=256)
    OVERLAP_SIZE: int = Field(default=128)
    CONFIDENCE_THRESHOLD: float = Field(default=0.7)
    
    # User modes for adaptive summarization
    USER_MODES: List[str] = Field(default=[
        "student", "researcher", "business", "beginner", "expert"
    ])
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("CONFIDENCE_THRESHOLD")
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure model cache directory exists
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)