"""
Summarization API Endpoints
Main endpoints for text, PDF, and YouTube summarization
"""

import asyncio
import logging
import time
from typing import Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
import json

from app.api.v1.models import (
    TextSummarizationRequest, TextSummarizationResponse,
    YouTubeSummarizationRequest, YouTubeSummarizationResponse,
    PDFSummarizationRequest, PDFSummarizationResponse,
    ErrorResponse
)
from app.services.summarization import HierarchicalSummarizer
from app.services.youtube import YouTubeProcessor
from app.services.pdf import PDFProcessor
from app.core.config import settings
from app.core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
summarizer = HierarchicalSummarizer()
youtube_processor = YouTubeProcessor()
pdf_processor = PDFProcessor()
rate_limiter = RateLimiter()

# Dependency for rate limiting
async def check_rate_limit():
    """Rate limiting dependency"""
    if not await rate_limiter.check_rate_limit("global", settings.RATE_LIMIT_PER_MINUTE):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

@router.post(
    "/text",
    response_model=TextSummarizationResponse,
    summary="Summarize Text Content",
    description="Generate intelligent summaries from raw text with optional query focus"
)
async def summarize_text(
    request: TextSummarizationRequest,
    _: None = Depends(check_rate_limit)
) -> TextSummarizationResponse:
    """
    Summarize text content using hierarchical map-reduce approach
    
    - **content**: Text content to summarize (10-1M characters)
    - **query**: Optional focus query for targeted summarization
    - **mode**: User mode (student, researcher, business, beginner, expert)
    - **max_length**: Maximum summary length (20-500 tokens)
    - **min_length**: Minimum summary length (10-200 tokens)
    - **use_cache**: Whether to use intelligent caching
    
    Returns comprehensive summary with confidence scores and explainability.
    """
    
    try:
        logger.info(f"Text summarization request: {len(request.content)} chars, mode: {request.mode}")
        
        # Validate content length
        if len(request.content) > settings.MAX_CONTENT_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=f"Content too large. Maximum size: {settings.MAX_CONTENT_LENGTH} characters"
            )
        
        # Process summarization
        result = await summarizer.summarize(
            content=request.content,
            content_type="text",
            query=request.query,
            mode=request.mode.value,
            max_length=request.max_length,
            min_length=request.min_length,
            use_cache=request.use_cache
        )
        
        # Convert to response model
        response = TextSummarizationResponse(
            summary_short=result.summary_short,
            summary_medium=result.summary_medium,
            summary_detailed=result.summary_detailed,
            key_points=result.key_points,
            query_focused_summary=result.query_focused_summary,
            timestamps=result.timestamps,
            confidence_scores=result.confidence_scores,
            citations=result.citations,
            explainability=result.explainability,
            processing_time=result.processing_time,
            models_used=result.models_used
        )
        
        logger.info(f"Text summarization completed in {result.processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text summarization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )

@router.post(
    "/youtube",
    response_model=YouTubeSummarizationResponse,
    summary="Summarize YouTube Video",
    description="Extract and summarize YouTube video content with timestamp awareness"
)
async def summarize_youtube(
    request: YouTubeSummarizationRequest,
    _: None = Depends(check_rate_limit)
) -> YouTubeSummarizationResponse:
    """
    Summarize YouTube video with advanced features:
    
    - **url**: YouTube video URL (youtube.com or youtu.be)
    - **query**: Optional focus query for targeted summarization
    - **mode**: User mode for adaptive summarization
    - **use_cache**: Whether to use intelligent caching
    
    Features:
    - Automatic transcript extraction (YouTube API + Whisper fallback)
    - Topic segmentation with timestamps
    - Skippable content detection (ads, repetition)
    - Timestamp-aware summaries
    """
    
    try:
        logger.info(f"YouTube summarization request: {request.url}, mode: {request.mode}")
        
        # Process YouTube video
        result = await youtube_processor.process_video(
            url=request.url,
            query=request.query,
            mode=request.mode.value,
            use_cache=request.use_cache
        )
        
        # Convert to response model
        response = YouTubeSummarizationResponse(
            video_info=result["video_info"],
            summary_short=result["summary_short"],
            summary_medium=result["summary_medium"],
            summary_detailed=result["summary_detailed"],
            key_points=result["key_points"],
            query_focused_summary=result["query_focused_summary"],
            timestamps=result["timestamps"],
            timestamped_summaries=result["timestamped_summaries"],
            topic_segments=result["topic_segments"],
            confidence_scores=result["confidence_scores"],
            citations=result["citations"],
            explainability=result["explainability"],
            processing_time=result["processing_time"],
            models_used=result["models_used"]
        )
        
        logger.info(f"YouTube summarization completed in {result['processing_time']:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YouTube summarization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"YouTube processing failed: {str(e)}"
        )

@router.post(
    "/pdf",
    response_model=PDFSummarizationResponse,
    summary="Summarize PDF Document",
    description="Extract and summarize PDF documents with structure preservation"
)
async def summarize_pdf(
    file: UploadFile = File(..., description="PDF file to summarize"),
    query: str = None,
    mode: str = "researcher",
    use_cache: bool = True,
    _: None = Depends(check_rate_limit)
) -> PDFSummarizationResponse:
    """
    Summarize PDF document with advanced features:
    
    - **file**: PDF file upload (max size based on settings)
    - **query**: Optional focus query for targeted summarization
    - **mode**: User mode (student, researcher, business, beginner, expert)
    - **use_cache**: Whether to use intelligent caching
    
    Features:
    - Multi-library text extraction (PyMuPDF + PyPDF2 fallback)
    - Document structure detection (headings, sections)
    - Page-aware citation mapping
    - Table and image detection
    - Section-based summaries
    """
    
    try:
        logger.info(f"PDF summarization request: {file.filename}, mode: {mode}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF document"
            )
        
        # Read file content
        pdf_content = await file.read()
        
        # Validate file size
        if len(pdf_content) > settings.MAX_CONTENT_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_CONTENT_LENGTH} bytes"
            )
        
        # Process PDF
        result = await pdf_processor.process_pdf(
            pdf_content=pdf_content,
            filename=file.filename,
            query=query,
            mode=mode,
            use_cache=use_cache
        )
        
        # Convert to response model
        response = PDFSummarizationResponse(
            document_info=result["document_info"],
            summary_short=result["summary_short"],
            summary_medium=result["summary_medium"],
            summary_detailed=result["summary_detailed"],
            key_points=result["key_points"],
            query_focused_summary=result["query_focused_summary"],
            timestamps=result["timestamps"],
            page_summaries=result["page_summaries"],
            section_summaries=result["section_summaries"],
            document_structure=result["document_structure"],
            confidence_scores=result["confidence_scores"],
            citations=result["citations"],
            explainability=result["explainability"],
            processing_time=result["processing_time"],
            models_used=result["models_used"]
        )
        
        logger.info(f"PDF summarization completed in {result['processing_time']:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF summarization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )

@router.post(
    "/text/stream",
    summary="Stream Text Summarization",
    description="Stream summarization results in real-time using Server-Sent Events"
)
async def stream_text_summarization(
    request: TextSummarizationRequest,
    _: None = Depends(check_rate_limit)
):
    """
    Stream text summarization results in real-time
    
    Returns Server-Sent Events (SSE) stream with:
    - Progress updates
    - Intermediate results
    - Final summary
    """
    
    async def generate_stream():
        try:
            yield f"data: {json.dumps({'status': 'started', 'message': 'Processing text...'})}\n\n"
            
            # Simulate streaming by yielding progress updates
            yield f"data: {json.dumps({'status': 'progress', 'step': 'chunking', 'progress': 20})}\n\n"
            
            await asyncio.sleep(0.1)  # Small delay for demo
            
            yield f"data: {json.dumps({'status': 'progress', 'step': 'summarizing', 'progress': 60})}\n\n"
            
            # Process actual summarization
            result = await summarizer.summarize(
                content=request.content,
                content_type="text",
                query=request.query,
                mode=request.mode.value,
                max_length=request.max_length,
                min_length=request.min_length,
                use_cache=request.use_cache
            )
            
            yield f"data: {json.dumps({'status': 'progress', 'step': 'finalizing', 'progress': 90})}\n\n"
            
            # Send final result
            response_data = {
                'status': 'completed',
                'result': {
                    'summary_short': result.summary_short,
                    'summary_medium': result.summary_medium,
                    'summary_detailed': result.summary_detailed,
                    'key_points': result.key_points,
                    'query_focused_summary': result.query_focused_summary,
                    'confidence_scores': result.confidence_scores,
                    'processing_time': result.processing_time
                }
            }
            
            yield f"data: {json.dumps(response_data)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {
                'status': 'error',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )