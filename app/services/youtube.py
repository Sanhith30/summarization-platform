"""
Advanced YouTube Video Processing Pipeline
Transcript extraction, topic segmentation, and timestamp-aware summarization
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import tempfile
import os

# Optional whisper import
try:
    import openai_whisper as whisper
    WHISPER_AVAILABLE = True
except ImportError:
    try:
        # Try alternative whisper package
        WHISPER_AVAILABLE = False
    except:
        WHISPER_AVAILABLE = False

from app.core.config import settings
from app.core.cache import CacheManager
from app.core.database import DatabaseManager
from app.services.summarization import HierarchicalSummarizer, SummarizationResult

logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    """YouTube transcript segment with timing"""
    text: str
    start: float
    duration: float
    end: float
    
    def __post_init__(self):
        if self.end == 0:
            self.end = self.start + self.duration

@dataclass
class VideoInfo:
    """YouTube video metadata"""
    video_id: str
    title: str
    duration: int  # seconds
    description: str
    upload_date: str
    view_count: int
    channel: str

@dataclass
class TopicSegment:
    """Video topic segment with timestamps"""
    topic: str
    start_time: float
    end_time: float
    text: str
    importance: str  # "high", "medium", "low", "skippable"
    summary: str = ""

class YouTubeProcessor:
    """
    Advanced YouTube video processing with:
    - Multi-source transcript extraction
    - Whisper fallback for audio
    - Topic segmentation with timestamps
    - Intelligent content filtering
    - Timestamp-aware summarization
    """
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.summarizer = HierarchicalSummarizer()
        self.whisper_model = None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats"""
        
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    async def process_video(
        self,
        url: str,
        query: Optional[str] = None,
        mode: str = "researcher",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Complete YouTube video processing pipeline
        """
        start_time = time.time()
        
        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            logger.info(f"Processing YouTube video: {video_id}")
            
            # Check cache first
            if use_cache:
                cached_result = await self.cache_manager.get_youtube_transcript(video_id)
                if cached_result and 'summary' in cached_result:
                    logger.info("Returning cached YouTube summary")
                    return cached_result
            
            # Step 1: Get video metadata
            video_info = await self._get_video_info(video_id)
            
            # Check duration limit
            if video_info.duration > settings.YOUTUBE_MAX_DURATION:
                raise ValueError(f"Video too long: {video_info.duration}s (max: {settings.YOUTUBE_MAX_DURATION}s)")
            
            # Step 2: Extract transcript
            transcript_segments = await self._extract_transcript(video_id, video_info)
            
            # Step 3: Process transcript into text
            full_text = self._segments_to_text(transcript_segments)
            
            # Step 4: Topic segmentation with timestamps
            topic_segments = await self._segment_by_topics(transcript_segments)
            
            # Step 5: Filter content (remove ads, repetition, etc.)
            filtered_segments = self._filter_content(topic_segments)
            
            # Step 6: Generate summaries
            summary_result = await self.summarizer.summarize(
                content=full_text,
                content_type="youtube",
                query=query,
                mode=mode,
                use_cache=False  # We handle caching at video level
            )
            
            # Step 7: Generate timestamp-aware summaries
            timestamped_summaries = await self._generate_timestamped_summaries(
                filtered_segments, mode
            )
            
            # Step 8: Create final result
            result = {
                "video_info": {
                    "video_id": video_id,
                    "title": video_info.title,
                    "duration": video_info.duration,
                    "channel": video_info.channel,
                    "url": url
                },
                "summary_short": summary_result.summary_short,
                "summary_medium": summary_result.summary_medium,
                "summary_detailed": summary_result.summary_detailed,
                "key_points": summary_result.key_points,
                "query_focused_summary": summary_result.query_focused_summary,
                "timestamps": [self._format_timestamp_range(seg.start_time, seg.end_time) for seg in filtered_segments],
                "timestamped_summaries": timestamped_summaries,
                "topic_segments": [
                    {
                        "topic": seg.topic,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "timestamp": self._format_timestamp_range(seg.start_time, seg.end_time),
                        "importance": seg.importance,
                        "summary": seg.summary
                    }
                    for seg in filtered_segments
                ],
                "confidence_scores": summary_result.confidence_scores,
                "citations": [f"Video: {video_info.title}"],
                "explainability": {
                    **summary_result.explainability,
                    "video_processing": {
                        "transcript_source": "youtube_api",
                        "segments_detected": len(topic_segments),
                        "segments_filtered": len(filtered_segments),
                        "duration_processed": video_info.duration
                    }
                },
                "processing_time": time.time() - start_time,
                "models_used": summary_result.models_used
            }
            
            # Cache result
            if use_cache:
                await self.cache_manager.set_youtube_transcript(
                    video_id, result, ttl=86400  # 24 hours
                )
            
            # Save to database
            await DatabaseManager.save_youtube_video(
                video_id=video_id,
                title=video_info.title,
                duration=video_info.duration,
                transcript_available=True,
                transcript_language="en",
                transcript_source="youtube_api"
            )
            
            logger.info(f"YouTube processing completed in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"YouTube processing failed: {e}")
            raise
    
    async def _get_video_info(self, video_id: str) -> VideoInfo:
        """Extract video metadata using yt-dlp"""
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
                )
            
            return VideoInfo(
                video_id=video_id,
                title=info.get('title', 'Unknown Title'),
                duration=info.get('duration', 0),
                description=info.get('description', ''),
                upload_date=info.get('upload_date', ''),
                view_count=info.get('view_count', 0),
                channel=info.get('uploader', 'Unknown Channel')
            )
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            # Fallback with minimal info
            return VideoInfo(
                video_id=video_id,
                title="Unknown Title",
                duration=0,
                description="",
                upload_date="",
                view_count=0,
                channel="Unknown Channel"
            )
    
    async def _extract_transcript(self, video_id: str, video_info: VideoInfo) -> List[TranscriptSegment]:
        """Extract transcript with fallback to Whisper"""
        
        try:
            # New API: instantiate YouTubeTranscriptApi
            api = YouTubeTranscriptApi()
            
            # Fetch transcript directly
            transcript_data = api.fetch(video_id)
            
            segments = []
            for item in transcript_data:
                segment = TranscriptSegment(
                    text=item.text,
                    start=item.start,
                    duration=item.duration,
                    end=item.start + item.duration
                )
                segments.append(segment)
            
            logger.info(f"Extracted {len(segments)} transcript segments from YouTube API")
            return segments
            
        except Exception as e:
            logger.warning(f"YouTube transcript not available: {e}")
            
            # Return empty if Whisper not available
            if not WHISPER_AVAILABLE:
                raise ValueError(f"Could not extract transcript: {e}")
            
            # Fallback to Whisper
            return await self._extract_with_whisper(video_id, video_info)
    
    async def _extract_with_whisper(self, video_id: str, video_info: VideoInfo) -> List[TranscriptSegment]:
        """Extract transcript using Whisper as fallback"""
        
        try:
            logger.info("Falling back to Whisper for transcript extraction")
            
            # Load Whisper model if not loaded
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model(settings.WHISPER_MODEL)
            
            # Download audio using yt-dlp
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
                
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': audio_path,
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: ydl.download([f"https://youtube.com/watch?v={video_id}"])
                    )
                
                # Transcribe with Whisper
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.whisper_model.transcribe(audio_path, word_timestamps=True)
                )
                
                # Convert to segments
                segments = []
                for segment in result['segments']:
                    transcript_segment = TranscriptSegment(
                        text=segment['text'].strip(),
                        start=segment['start'],
                        duration=segment['end'] - segment['start'],
                        end=segment['end']
                    )
                    segments.append(transcript_segment)
                
                logger.info(f"Extracted {len(segments)} segments using Whisper")
                return segments
                
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise ValueError("Could not extract transcript from video")
    
    def _segments_to_text(self, segments: List[TranscriptSegment]) -> str:
        """Convert transcript segments to continuous text"""
        return " ".join([seg.text for seg in segments])
    
    async def _segment_by_topics(self, transcript_segments: List[TranscriptSegment]) -> List[TopicSegment]:
        """Segment video by topics using sliding window approach"""
        
        if not transcript_segments:
            return []
        
        # Group segments into larger chunks (30-60 seconds)
        topic_segments = []
        current_text = ""
        current_start = transcript_segments[0].start
        current_end = transcript_segments[0].end
        
        for i, segment in enumerate(transcript_segments):
            current_text += " " + segment.text
            current_end = segment.end
            
            # Check if we should create a new topic segment
            duration = current_end - current_start
            
            if duration >= 30 or i == len(transcript_segments) - 1:  # 30 second chunks or last segment
                # Detect topic (simplified - could use more advanced NLP)
                topic = self._detect_topic(current_text)
                
                topic_segment = TopicSegment(
                    topic=topic,
                    start_time=current_start,
                    end_time=current_end,
                    text=current_text.strip(),
                    importance="medium"  # Will be refined later
                )
                
                topic_segments.append(topic_segment)
                
                # Start next segment
                if i < len(transcript_segments) - 1:
                    current_text = ""
                    current_start = segment.end
        
        # Classify importance
        topic_segments = self._classify_segment_importance(topic_segments)
        
        return topic_segments
    
    def _detect_topic(self, text: str) -> str:
        """Simple topic detection based on keywords"""
        
        text_lower = text.lower()
        
        # Common video section patterns
        if any(word in text_lower for word in ['introduction', 'intro', 'welcome', 'hello']):
            return "Introduction"
        elif any(word in text_lower for word in ['conclusion', 'summary', 'wrap up', 'final']):
            return "Conclusion"
        elif any(word in text_lower for word in ['example', 'demo', 'demonstration']):
            return "Example/Demo"
        elif any(word in text_lower for word in ['question', 'q&a', 'answer']):
            return "Q&A"
        elif any(word in text_lower for word in ['sponsor', 'advertisement', 'ad', 'promo']):
            return "Advertisement"
        else:
            # Extract key terms for topic
            words = text_lower.split()
            # Simple heuristic: find most frequent meaningful words
            meaningful_words = [w for w in words if len(w) > 4 and w.isalpha()]
            if meaningful_words:
                return f"Topic: {meaningful_words[0].title()}"
            else:
                return "General Content"
    
    def _classify_segment_importance(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        """Classify segment importance for skippability"""
        
        for segment in segments:
            text_lower = segment.text.lower()
            
            # High importance indicators
            if any(word in text_lower for word in [
                'important', 'key', 'main', 'core', 'essential', 'critical'
            ]):
                segment.importance = "high"
            
            # Low importance / skippable indicators
            elif any(word in text_lower for word in [
                'sponsor', 'advertisement', 'ad', 'promo', 'like and subscribe',
                'patreon', 'merch', 'social media'
            ]):
                segment.importance = "skippable"
            
            # Repetitive content detection
            elif self._is_repetitive(segment.text):
                segment.importance = "low"
            
            # Default to medium
            else:
                segment.importance = "medium"
        
        return segments
    
    def _is_repetitive(self, text: str) -> bool:
        """Detect repetitive content"""
        
        sentences = text.split('.')
        if len(sentences) < 3:
            return False
        
        # Check for repeated phrases
        words = text.lower().split()
        word_count = {}
        for word in words:
            if len(word) > 3:
                word_count[word] = word_count.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, consider repetitive
        max_frequency = max(word_count.values()) if word_count else 0
        return max_frequency > len(words) * 0.3
    
    def _filter_content(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        """Filter out low-value content"""
        
        filtered = []
        
        for segment in segments:
            # Skip very short segments (less than 10 seconds)
            if segment.end_time - segment.start_time < 10:
                continue
            
            # Skip skippable content unless it's the only content
            if segment.importance == "skippable" and len(segments) > 1:
                continue
            
            # Skip if text is too short
            if len(segment.text.split()) < 10:
                continue
            
            filtered.append(segment)
        
        return filtered
    
    async def _generate_timestamped_summaries(
        self, 
        segments: List[TopicSegment], 
        mode: str
    ) -> List[Dict[str, Any]]:
        """Generate summaries for each timestamp segment"""
        
        timestamped_summaries = []
        
        for segment in segments:
            try:
                # Generate summary for this segment
                summary_result = await self.summarizer.summarize(
                    content=segment.text,
                    content_type="youtube_segment",
                    mode=mode,
                    use_cache=False
                )
                
                segment.summary = summary_result.summary_short
                
                timestamped_summary = {
                    "timestamp": self._format_timestamp_range(segment.start_time, segment.end_time),
                    "topic": segment.topic,
                    "summary": segment.summary,
                    "importance": segment.importance,
                    "duration": segment.end_time - segment.start_time
                }
                
                timestamped_summaries.append(timestamped_summary)
                
            except Exception as e:
                logger.error(f"Failed to summarize segment: {e}")
                continue
        
        return timestamped_summaries
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS"""
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _format_timestamp_range(self, start: float, end: float) -> str:
        """Format timestamp range"""
        return f"{self._format_timestamp(start)}–{self._format_timestamp(end)}"