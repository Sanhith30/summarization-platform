"""
Advanced Summarization Engine
Hierarchical Map-Reduce with Query-Focused Intelligence
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.models import ModelManager, SummaryResult
from app.core.config import settings
from app.core.cache import CacheManager
from app.core.database import DatabaseManager

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

@dataclass
class TextChunk:
    """Text chunk with metadata"""
    text: str
    start_pos: int
    end_pos: int
    topic_score: float = 0.0
    relevance_score: float = 0.0
    confidence: float = 0.0

@dataclass
class SummarizationResult:
    """Complete summarization result"""
    summary_short: str
    summary_medium: str
    summary_detailed: str
    key_points: List[str]
    query_focused_summary: str
    timestamps: List[str]
    confidence_scores: List[float]
    citations: List[str]
    explainability: Dict[str, Any]
    processing_time: float
    models_used: List[str]

class AdaptiveChunker:
    """
    Intelligent text chunking with:
    - Token-aware segmentation
    - Topic boundary detection
    - Overlap management
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def chunk_text(
        self, 
        text: str, 
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        overlap_size: int = None
    ) -> List[TextChunk]:
        """Create intelligent text chunks"""
        
        max_size = max_chunk_size or settings.MAX_CHUNK_SIZE
        min_size = min_chunk_size or settings.MIN_CHUNK_SIZE
        overlap = overlap_size or settings.OVERLAP_SIZE
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding sentence exceeds max size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk.split()) > max_size and current_chunk:
                # Create chunk from current content
                if len(current_chunk.split()) >= min_size:
                    chunk = TextChunk(
                        text=current_chunk.strip(),
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) if overlap_text else current_start + len(current_chunk)
            else:
                current_chunk = potential_chunk
        
        # Add final chunk - use lower threshold for final chunk to avoid losing content
        final_min = min(min_size, 10)  # At least 10 words for final chunk
        if current_chunk and len(current_chunk.split()) >= final_min:
            chunk = TextChunk(
                text=current_chunk.strip(),
                start_pos=current_start,
                end_pos=current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of chunk"""
        words = text.split()
        if len(words) <= overlap_size:
            return text
        return " ".join(words[-overlap_size:])
    
    def detect_topic_boundaries(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Detect topic shifts and score chunks"""
        if len(chunks) < 2:
            return chunks
        
        # Calculate TF-IDF for each chunk
        chunk_texts = [chunk.text for chunk in chunks]
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(chunk_texts)
            
            # Calculate similarity between adjacent chunks
            for i in range(len(chunks)):
                if i == 0:
                    chunks[i].topic_score = 1.0  # First chunk
                else:
                    # Similarity with previous chunk
                    similarity = cosine_similarity(
                        tfidf_matrix[i-1:i], 
                        tfidf_matrix[i:i+1]
                    )[0][0]
                    
                    # Lower similarity = higher topic shift score
                    chunks[i].topic_score = 1.0 - similarity
            
        except Exception as e:
            logger.warning(f"Topic detection failed: {e}")
            # Fallback: uniform scoring
            for chunk in chunks:
                chunk.topic_score = 0.5
        
        return chunks

class QueryFocusedProcessor:
    """
    Query-focused summarization with relevance scoring
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def score_relevance(
        self, 
        chunks: List[TextChunk], 
        query: Optional[str] = None
    ) -> List[TextChunk]:
        """Score chunk relevance to query"""
        
        if not query:
            # No query - uniform relevance
            for chunk in chunks:
                chunk.relevance_score = 1.0
            return chunks
        
        try:
            # Prepare texts
            chunk_texts = [chunk.text for chunk in chunks]
            all_texts = chunk_texts + [query]
            
            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[-1:]  # Last item is query
            chunk_vectors = tfidf_matrix[:-1]  # All except query
            
            # Calculate similarities
            similarities = cosine_similarity(chunk_vectors, query_vector).flatten()
            
            # Normalize scores
            if similarities.max() > 0:
                similarities = similarities / similarities.max()
            
            # Assign relevance scores
            for i, chunk in enumerate(chunks):
                chunk.relevance_score = float(similarities[i])
            
        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            # Fallback: uniform scoring
            for chunk in chunks:
                chunk.relevance_score = 1.0
        
        return chunks
    
    def filter_relevant_chunks(
        self, 
        chunks: List[TextChunk], 
        threshold: float = 0.3,
        max_chunks: int = 10
    ) -> List[TextChunk]:
        """Filter chunks by relevance threshold"""
        
        # Sort by relevance score
        relevant_chunks = [
            chunk for chunk in chunks 
            if chunk.relevance_score >= threshold
        ]
        
        # Sort by relevance (descending)
        relevant_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit number of chunks
        return relevant_chunks[:max_chunks]

class HierarchicalSummarizer:
    """
    Map-Reduce hierarchical summarization with ensemble support
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.cache_manager = CacheManager()
        self.chunker = AdaptiveChunker()
        self.query_processor = QueryFocusedProcessor()
    
    async def summarize(
        self,
        content: str,
        content_type: str = "text",
        query: Optional[str] = None,
        mode: str = "researcher",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        use_cache: bool = True
    ) -> SummarizationResult:
        """
        Main summarization pipeline with hierarchical processing
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache:
                cached_result = await self.cache_manager.get_summary(
                    content, query=query, mode=mode, 
                    max_length=max_length, min_length=min_length
                )
                if cached_result:
                    logger.info("Returning cached summary")
                    return SummarizationResult(**cached_result)
            
            # Step 1: Adaptive chunking
            logger.info("Starting hierarchical summarization...")
            chunks = self.chunker.chunk_text(content)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Handle short texts that don't get chunked
            if not chunks and content.strip():
                # Create a single chunk for short content
                chunks = [TextChunk(
                    text=content.strip(),
                    start_pos=0,
                    end_pos=len(content),
                    topic_score=1.0,
                    relevance_score=1.0
                )]
                logger.info("Created single chunk for short content")
            
            # Step 2: Topic segmentation
            chunks = self.chunker.detect_topic_boundaries(chunks)
            
            # Step 3: Query-focused filtering (if query provided)
            if query:
                chunks = self.query_processor.score_relevance(chunks, query)
                chunks = self.query_processor.filter_relevant_chunks(chunks)
                logger.info(f"Filtered to {len(chunks)} relevant chunks")
            
            # Step 4: Map phase - summarize chunks in parallel
            chunk_summaries = await self._map_phase(chunks, mode)
            
            # Step 5: Reduce phase - hierarchical merging
            final_summaries = await self._reduce_phase(chunk_summaries, mode)
            
            # Step 6: Generate different summary lengths
            summary_variants = await self._generate_summary_variants(
                final_summaries, content, mode
            )
            
            # Step 7: Extract key points
            key_points = await self._extract_key_points(content, chunks)
            
            # Step 8: Query-focused summary (if query provided)
            query_focused = ""
            if query:
                query_focused = await self._generate_query_focused_summary(
                    content, query, chunks
                )
            
            # Step 9: Confidence scoring and explainability
            confidence_scores, explainability = self._calculate_confidence_and_explainability(
                chunk_summaries, final_summaries, chunks
            )
            
            # Step 10: Generate citations
            citations = self._generate_citations(chunks, content_type)
            
            processing_time = time.time() - start_time
            models_used = list(set([s.model_used for s in chunk_summaries]))
            
            # Create result
            result = SummarizationResult(
                summary_short=summary_variants.get("short", ""),
                summary_medium=summary_variants.get("medium", ""),
                summary_detailed=summary_variants.get("detailed", ""),
                key_points=key_points,
                query_focused_summary=query_focused,
                timestamps=[],  # Will be populated for video content
                confidence_scores=confidence_scores,
                citations=citations,
                explainability=explainability,
                processing_time=processing_time,
                models_used=models_used
            )
            
            # Cache result
            if use_cache:
                await self.cache_manager.set_summary(
                    content, result.__dict__, 
                    query=query, mode=mode,
                    max_length=max_length, min_length=min_length
                )
            
            # Save to database
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            await DatabaseManager.save_summarization_request(
                content_hash=content_hash,
                content_type=content_type,
                content_length=len(content),
                summary_data=result.__dict__,
                processing_time=processing_time,
                models_used=models_used,
                cache_hit=False,
                query=query,
                mode=mode,
                max_length=max_length,
                min_length=min_length
            )
            
            logger.info(f"Summarization completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise
    
    async def _map_phase(
        self, 
        chunks: List[TextChunk], 
        mode: str
    ) -> List[SummaryResult]:
        """Map phase: summarize chunks in parallel"""
        
        if not chunks:
            return []
        
        # Adjust parameters based on mode
        max_len, min_len = self._get_mode_parameters(mode)
        
        # Create summarization tasks
        tasks = []
        for chunk in chunks:
            task = self.model_manager.summarize_text(
                chunk.text,
                max_length=max_len,
                min_length=min_len,
                use_ensemble=False  # Disabled for speed - single model is faster
            )
            tasks.append(task)
        
        # Execute in parallel with concurrency limit
        semaphore = asyncio.Semaphore(3)  # Limit concurrent tasks
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[bounded_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        chunk_summaries = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Chunk {i} summarization failed: {result}")
                continue
            
            if isinstance(result, list):
                # Ensemble result - take best
                best_result = max(result, key=lambda x: x.confidence)
                chunk_summaries.append(best_result)
            else:
                chunk_summaries.append(result)
        
        return chunk_summaries
    
    async def _reduce_phase(
        self, 
        chunk_summaries: List[SummaryResult], 
        mode: str
    ) -> List[SummaryResult]:
        """Reduce phase: hierarchical merging of summaries"""
        
        if len(chunk_summaries) <= 1:
            return chunk_summaries
        
        current_summaries = chunk_summaries
        
        # Hierarchical reduction
        while len(current_summaries) > 3:
            # Group summaries for merging
            groups = [
                current_summaries[i:i+3] 
                for i in range(0, len(current_summaries), 3)
            ]
            
            # Merge each group
            merge_tasks = []
            for group in groups:
                combined_text = " ".join([s.text for s in group])
                max_len, min_len = self._get_mode_parameters(mode, is_reduce=True)
                
                task = self.model_manager.summarize_text(
                    combined_text,
                    max_length=max_len,
                    min_length=min_len,
                    use_ensemble=False  # Single model for reduce phase
                )
                merge_tasks.append(task)
            
            # Execute merging
            merge_results = await asyncio.gather(*merge_tasks)
            
            # Update current summaries
            current_summaries = []
            for result in merge_results:
                if isinstance(result, list):
                    current_summaries.append(result[0])
                else:
                    current_summaries.append(result)
        
        return current_summaries
    
    async def _generate_summary_variants(
        self, 
        final_summaries: List[SummaryResult], 
        original_content: str,
        mode: str
    ) -> Dict[str, str]:
        """Generate short, medium, and detailed summaries"""
        
        if not final_summaries:
            return {"short": "", "medium": "", "detailed": ""}
        
        # Combine final summaries
        combined_summary = " ".join([s.text for s in final_summaries])
        
        # Generate variants with different lengths
        variants = {}
        
        # Short summary (1-2 sentences)
        short_result = await self.model_manager.summarize_text(
            combined_summary,
            max_length=50,
            min_length=20,
            use_ensemble=False
        )
        variants["short"] = short_result[0].text if short_result else ""
        
        # Medium summary (3-5 sentences)
        medium_result = await self.model_manager.summarize_text(
            combined_summary,
            max_length=100,
            min_length=60,
            use_ensemble=False
        )
        variants["medium"] = medium_result[0].text if medium_result else ""
        
        # Detailed summary (use combined summary)
        variants["detailed"] = combined_summary
        
        return variants
    
    async def _extract_key_points(
        self, 
        content: str, 
        chunks: List[TextChunk]
    ) -> List[str]:
        """Extract key points from content"""
        
        try:
            # Use highest relevance chunks for key points
            top_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)[:5]
            
            key_points = []
            for chunk in top_chunks:
                # Extract most important sentence from chunk
                sentences = sent_tokenize(chunk.text)
                if sentences:
                    # Simple heuristic: longest sentence often contains key info
                    key_sentence = max(sentences, key=len)
                    if len(key_sentence.split()) > 5:  # Minimum length
                        key_points.append(key_sentence.strip())
            
            return key_points[:5]  # Limit to 5 key points
            
        except Exception as e:
            logger.error(f"Key point extraction failed: {e}")
            return []
    
    async def _generate_query_focused_summary(
        self, 
        content: str, 
        query: str, 
        chunks: List[TextChunk]
    ) -> str:
        """Generate summary focused on specific query"""
        
        try:
            # Get most relevant chunks
            relevant_chunks = [
                chunk for chunk in chunks 
                if chunk.relevance_score > 0.5
            ]
            
            if not relevant_chunks:
                return ""
            
            # Combine relevant content
            relevant_text = " ".join([chunk.text for chunk in relevant_chunks])
            
            # Create query-focused prompt
            focused_text = f"Query: {query}\n\nRelevant content: {relevant_text}"
            
            # Generate focused summary
            result = await self.model_manager.summarize_text(
                focused_text,
                max_length=150,
                min_length=50,
                use_ensemble=False
            )
            
            return result[0].text if result else ""
            
        except Exception as e:
            logger.error(f"Query-focused summary failed: {e}")
            return ""
    
    def _calculate_confidence_and_explainability(
        self,
        chunk_summaries: List[SummaryResult],
        final_summaries: List[SummaryResult],
        chunks: List[TextChunk]
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Calculate confidence scores and explainability metrics"""
        
        # Calculate confidence scores
        confidence_scores = []
        if chunk_summaries:
            confidence_scores = [s.confidence for s in chunk_summaries]
        
        # Overall confidence (average)
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.7
        
        # Explainability information
        explainability = {
            "method": "hierarchical_map_reduce",
            "confidence": float(overall_confidence),
            "source_alignment": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.6 else "low",
            "chunk_count": len(chunks),
            "models_used": len(set([s.model_used for s in chunk_summaries])) if chunk_summaries else 1,
            "processing_stages": [
                "adaptive_chunking",
                "topic_segmentation", 
                "relevance_scoring",
                "parallel_summarization",
                "hierarchical_reduction",
                "confidence_validation"
            ]
        }
        
        return confidence_scores, explainability
    
    def _generate_citations(
        self, 
        chunks: List[TextChunk], 
        content_type: str
    ) -> List[str]:
        """Generate source citations"""
        
        citations = []
        
        for i, chunk in enumerate(chunks):
            if content_type == "youtube":
                # For videos, use timestamp-based citations
                citations.append(f"Video segment {i+1}")
            elif content_type == "pdf":
                # For PDFs, use page-based citations
                citations.append(f"Document section {i+1}")
            else:
                # For text, use position-based citations
                citations.append(f"Text section {i+1} (chars {chunk.start_pos}-{chunk.end_pos})")
        
        return citations
    
    def _get_mode_parameters(self, mode: str, is_reduce: bool = False) -> Tuple[int, int]:
        """Get summarization parameters based on user mode"""
        
        base_params = {
            "student": (80, 30),
            "researcher": (120, 50),
            "business": (100, 40),
            "beginner": (60, 25),
            "expert": (150, 60)
        }
        
        max_len, min_len = base_params.get(mode, (100, 40))
        
        # Adjust for reduce phase
        if is_reduce:
            max_len = int(max_len * 1.2)
            min_len = int(min_len * 1.1)
        
        return max_len, min_len