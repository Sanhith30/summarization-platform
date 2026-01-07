"""
Pydantic models for API request/response validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class UserMode(str, Enum):
    """User modes for adaptive summarization"""
    STUDENT = "student"
    RESEARCHER = "researcher"
    BUSINESS = "business"
    BEGINNER = "beginner"
    EXPERT = "expert"

class ContentType(str, Enum):
    """Content types for processing"""
    TEXT = "text"
    PDF = "pdf"
    YOUTUBE = "youtube"

# Request Models
class TextSummarizationRequest(BaseModel):
    """Request model for text summarization"""
    content: str = Field(..., min_length=10, max_length=1000000, description="Text content to summarize")
    query: Optional[str] = Field(None, max_length=500, description="Optional focus query")
    mode: UserMode = Field(UserMode.RESEARCHER, description="User mode for adaptive summarization")
    max_length: Optional[int] = Field(None, ge=20, le=500, description="Maximum summary length")
    min_length: Optional[int] = Field(None, ge=10, le=200, description="Minimum summary length")
    use_cache: bool = Field(True, description="Whether to use caching")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @validator('min_length', 'max_length')
    def validate_lengths(cls, v, values):
        if 'min_length' in values and 'max_length' in values:
            if values['min_length'] and v and values['min_length'] >= v:
                raise ValueError('max_length must be greater than min_length')
        return v

class YouTubeSummarizationRequest(BaseModel):
    """Request model for YouTube video summarization"""
    url: str = Field(..., description="YouTube video URL")
    query: Optional[str] = Field(None, max_length=500, description="Optional focus query")
    mode: UserMode = Field(UserMode.RESEARCHER, description="User mode for adaptive summarization")
    use_cache: bool = Field(True, description="Whether to use caching")
    
    @validator('url')
    def validate_youtube_url(cls, v):
        if not any(domain in v.lower() for domain in ['youtube.com', 'youtu.be']):
            raise ValueError('Must be a valid YouTube URL')
        return v

class PDFSummarizationRequest(BaseModel):
    """Request model for PDF summarization"""
    query: Optional[str] = Field(None, max_length=500, description="Optional focus query")
    mode: UserMode = Field(UserMode.RESEARCHER, description="User mode for adaptive summarization")
    use_cache: bool = Field(True, description="Whether to use caching")

class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    request_id: int = Field(..., description="ID of the summarization request")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, max_length=1000, description="Optional feedback text")
    feedback_type: str = Field("quality", description="Type of feedback")
    user_mode: Optional[UserMode] = Field(None, description="User mode when feedback was given")

# Response Models
class VideoInfo(BaseModel):
    """YouTube video information"""
    video_id: str
    title: str
    duration: int
    channel: str
    url: str

class DocumentInfo(BaseModel):
    """PDF document information"""
    filename: str
    total_pages: int
    total_words: int
    has_structure: bool
    metadata: Dict[str, Any]

class TopicSegment(BaseModel):
    """Video topic segment"""
    topic: str
    start_time: float
    end_time: float
    timestamp: str
    importance: str
    summary: str

class PageSummary(BaseModel):
    """PDF page summary"""
    page_number: int
    word_count: int
    summary: str
    key_points: List[str]
    has_images: bool
    has_tables: bool

class SectionSummary(BaseModel):
    """PDF section summary"""
    title: str
    level: int
    pages: str
    word_count: int
    summary: str
    key_points: List[str]

class DocumentStructure(BaseModel):
    """PDF document structure"""
    title: str
    level: int
    pages: str
    word_count: int

class TimestampedSummary(BaseModel):
    """Timestamped summary for videos"""
    timestamp: str
    topic: str
    summary: str
    importance: str
    duration: float

class Explainability(BaseModel):
    """Explainability information"""
    method: str
    confidence: float
    source_alignment: str
    chunk_count: int
    models_used: int
    processing_stages: List[str]

class SummarizationResponse(BaseModel):
    """Base response model for all summarization types"""
    summary_short: str = Field(..., description="Brief summary (1-2 sentences)")
    summary_medium: str = Field(..., description="Medium summary (3-5 sentences)")
    summary_detailed: str = Field(..., description="Detailed summary (comprehensive)")
    key_points: List[str] = Field(..., description="Key points extracted")
    query_focused_summary: str = Field("", description="Query-focused summary if query provided")
    timestamps: List[str] = Field(default_factory=list, description="Timestamps for video content")
    confidence_scores: List[float] = Field(..., description="Confidence scores for each part")
    citations: List[str] = Field(..., description="Source citations")
    explainability: Explainability = Field(..., description="Explainability information")
    processing_time: float = Field(..., description="Processing time in seconds")
    models_used: List[str] = Field(..., description="AI models used")

class TextSummarizationResponse(SummarizationResponse):
    """Response model for text summarization"""
    pass

class YouTubeSummarizationResponse(SummarizationResponse):
    """Response model for YouTube video summarization"""
    video_info: VideoInfo = Field(..., description="Video metadata")
    timestamped_summaries: List[TimestampedSummary] = Field(..., description="Timestamped summaries")
    topic_segments: List[TopicSegment] = Field(..., description="Topic segments with timestamps")

class PDFSummarizationResponse(SummarizationResponse):
    """Response model for PDF summarization"""
    document_info: DocumentInfo = Field(..., description="Document metadata")
    page_summaries: List[PageSummary] = Field(..., description="Per-page summaries")
    section_summaries: List[SectionSummary] = Field(..., description="Section summaries")
    document_structure: List[DocumentStructure] = Field(..., description="Document structure")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    models: Dict[str, Any]
    cache: Dict[str, Any]
    version: str

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: float
    request_id: Optional[str] = None

class CacheStatsResponse(BaseModel):
    """Cache statistics response"""
    status: str
    memory_usage: Optional[str] = None
    total_keys: int = 0
    summary_keys: int = 0
    youtube_keys: int = 0
    hits: int = 0
    misses: int = 0

class ModelStatsResponse(BaseModel):
    """Model statistics response"""
    total_models: int
    loaded_models: int
    device: str
    models: Dict[str, Dict[str, Any]]

class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    success: bool
    feedback_id: int
    message: str