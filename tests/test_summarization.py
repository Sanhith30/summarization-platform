"""
Test suite for summarization functionality
Production-ready tests with fixtures and mocking
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.services.summarization import HierarchicalSummarizer, AdaptiveChunker, QueryFocusedProcessor
from app.core.models import ModelManager, SummaryResult
from app.core.cache import CacheManager

class TestAdaptiveChunker:
    """Test adaptive text chunking functionality"""
    
    def setup_method(self):
        self.chunker = AdaptiveChunker()
    
    def test_basic_chunking(self):
        """Test basic text chunking"""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = self.chunker.chunk_text(text, max_chunk_size=10, min_chunk_size=5)
        
        assert len(chunks) > 0
        assert all(chunk.text for chunk in chunks)
        assert all(chunk.start_pos >= 0 for chunk in chunks)
        assert all(chunk.end_pos > chunk.start_pos for chunk in chunks)
    
    def test_empty_text(self):
        """Test chunking with empty text"""
        chunks = self.chunker.chunk_text("")
        assert len(chunks) == 0
    
    def test_short_text(self):
        """Test chunking with very short text"""
        text = "Short text."
        chunks = self.chunker.chunk_text(text, min_chunk_size=20)
        # Should not create chunks if text is too short
        assert len(chunks) == 0 or len(chunks[0].text.split()) >= 20
    
    def test_topic_detection(self):
        """Test topic boundary detection"""
        chunks = [
            Mock(text="This is about machine learning and AI.", start_pos=0, end_pos=40),
            Mock(text="Now we discuss cooking and recipes.", start_pos=40, end_pos=75)
        ]
        
        result_chunks = self.chunker.detect_topic_boundaries(chunks)
        assert len(result_chunks) == len(chunks)
        assert all(hasattr(chunk, 'topic_score') for chunk in result_chunks)

class TestQueryFocusedProcessor:
    """Test query-focused processing"""
    
    def setup_method(self):
        self.processor = QueryFocusedProcessor()
    
    def test_relevance_scoring_with_query(self):
        """Test relevance scoring with a query"""
        chunks = [
            Mock(text="Machine learning is a subset of artificial intelligence."),
            Mock(text="Cooking pasta requires boiling water first."),
            Mock(text="Deep learning uses neural networks for AI tasks.")
        ]
        
        query = "artificial intelligence machine learning"
        result_chunks = self.processor.score_relevance(chunks, query)
        
        assert len(result_chunks) == len(chunks)
        assert all(hasattr(chunk, 'relevance_score') for chunk in result_chunks)
        
        # First and third chunks should have higher relevance
        assert result_chunks[0].relevance_score > result_chunks[1].relevance_score
        assert result_chunks[2].relevance_score > result_chunks[1].relevance_score
    
    def test_relevance_scoring_without_query(self):
        """Test relevance scoring without query (uniform scoring)"""
        chunks = [Mock(text="Some text"), Mock(text="Other text")]
        result_chunks = self.processor.score_relevance(chunks, None)
        
        assert all(chunk.relevance_score == 1.0 for chunk in result_chunks)
    
    def test_filter_relevant_chunks(self):
        """Test filtering chunks by relevance"""
        chunks = [
            Mock(relevance_score=0.9),
            Mock(relevance_score=0.2),
            Mock(relevance_score=0.7),
            Mock(relevance_score=0.1)
        ]
        
        filtered = self.processor.filter_relevant_chunks(chunks, threshold=0.5)
        assert len(filtered) == 2  # Only chunks with score >= 0.5
        assert all(chunk.relevance_score >= 0.5 for chunk in filtered)

class TestModelManager:
    """Test model management functionality"""
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager for testing"""
        manager = ModelManager()
        manager.models = {
            "test-model": Mock(),
        }
        manager.tokenizers = {
            "test-model": Mock(),
        }
        manager.model_info = {
            "test-model": Mock(
                name="test-model",
                model_type="test",
                max_length=100,
                min_length=20,
                is_loaded=True
            )
        }
        return manager
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_model_manager):
        """Test model health check"""
        health = await mock_model_manager.health_check()
        
        assert "total_models" in health
        assert "loaded_models" in health
        assert "device" in health
        assert "models" in health
    
    @pytest.mark.asyncio
    async def test_summarize_text_single_model(self, mock_model_manager):
        """Test single model text summarization"""
        with patch.object(mock_model_manager, '_single_model_summarize') as mock_summarize:
            mock_result = SummaryResult(
                text="Test summary",
                confidence=0.9,
                model_used="test-model",
                processing_time=1.0,
                token_count=10
            )
            mock_summarize.return_value = mock_result
            
            results = await mock_model_manager.summarize_text(
                "Test text", use_ensemble=False
            )
            
            assert len(results) == 1
            assert results[0].text == "Test summary"
            assert results[0].confidence == 0.9

class TestCacheManager:
    """Test cache management functionality"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager"""
        manager = CacheManager()
        manager.is_connected = True
        manager.redis_client = AsyncMock()
        return manager
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, mock_cache_manager):
        """Test cache key generation"""
        content = "Test content for caching"
        key1 = mock_cache_manager._generate_cache_key(content)
        key2 = mock_cache_manager._generate_cache_key(content)
        
        # Same content should generate same key
        assert key1 == key2
        
        # Different content should generate different keys
        key3 = mock_cache_manager._generate_cache_key("Different content")
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, mock_cache_manager):
        """Test cache set and get operations"""
        content = "Test content"
        summary_data = {"summary": "Test summary"}
        
        # Mock Redis operations
        mock_cache_manager.redis_client.get.return_value = None
        mock_cache_manager.redis_client.setex.return_value = True
        
        # Test cache miss
        result = await mock_cache_manager.get_summary(content)
        assert result is None
        
        # Test cache set
        await mock_cache_manager.set_summary(content, summary_data)
        mock_cache_manager.redis_client.setex.assert_called_once()

class TestHierarchicalSummarizer:
    """Test hierarchical summarization pipeline"""
    
    @pytest.fixture
    def mock_summarizer(self):
        """Create a mock summarizer with dependencies"""
        summarizer = HierarchicalSummarizer()
        
        # Mock dependencies
        summarizer.model_manager = AsyncMock()
        summarizer.cache_manager = AsyncMock()
        summarizer.chunker = Mock()
        summarizer.query_processor = Mock()
        
        return summarizer
    
    @pytest.mark.asyncio
    async def test_summarization_pipeline(self, mock_summarizer):
        """Test complete summarization pipeline"""
        # Setup mocks
        mock_chunks = [
            Mock(text="Chunk 1", relevance_score=0.9),
            Mock(text="Chunk 2", relevance_score=0.8)
        ]
        
        mock_summarizer.chunker.chunk_text.return_value = mock_chunks
        mock_summarizer.chunker.detect_topic_boundaries.return_value = mock_chunks
        mock_summarizer.query_processor.score_relevance.return_value = mock_chunks
        mock_summarizer.query_processor.filter_relevant_chunks.return_value = mock_chunks
        
        mock_summary_result = SummaryResult(
            text="Test summary",
            confidence=0.9,
            model_used="test-model",
            processing_time=1.0,
            token_count=10
        )
        
        mock_summarizer.model_manager.summarize_text.return_value = [mock_summary_result]
        mock_summarizer.cache_manager.get_summary.return_value = None
        
        # Test summarization
        result = await mock_summarizer.summarize(
            content="Test content for summarization",
            content_type="text"
        )
        
        assert result is not None
        assert hasattr(result, 'summary_short')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'models_used')

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_text_summarization_flow(self):
        """Test complete text summarization flow"""
        # This would test the actual API endpoint
        # In a real implementation, you'd use a test client
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in summarization pipeline"""
        summarizer = HierarchicalSummarizer()
        
        # Test with invalid input
        with pytest.raises(Exception):
            await summarizer.summarize(content="", content_type="text")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        # Test chunking performance
        chunker = AdaptiveChunker()
        large_text = "Test sentence. " * 1000
        
        import time
        start_time = time.time()
        chunks = chunker.chunk_text(large_text)
        end_time = time.time()
        
        # Should process 1000 sentences quickly
        assert end_time - start_time < 5.0  # Less than 5 seconds
        assert len(chunks) > 0

# Fixtures for test data
@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """

@pytest.fixture
def sample_long_text():
    """Long sample text for chunking tests"""
    return """
    Machine learning is a method of data analysis that automates analytical 
    model building. It is a branch of artificial intelligence based on the 
    idea that systems can learn from data, identify patterns and make decisions 
    with minimal human intervention. Deep learning is part of a broader family 
    of machine learning methods based on artificial neural networks with 
    representation learning. Learning can be supervised, semi-supervised or 
    unsupervised. The term "deep learning" was introduced to the machine 
    learning community by Rina Dechter in 1986, and to artificial neural 
    networks by Igor Aizenberg and colleagues in 2000, in the context of 
    Boolean threshold neurons.
    """ * 10  # Repeat to make it longer

if __name__ == "__main__":
    pytest.main([__file__, "-v"])