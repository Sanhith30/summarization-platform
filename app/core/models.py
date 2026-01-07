"""
AI Model Management System
Handles model loading, ensemble, and inference with fallback support
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    BartForConditionalGeneration, BartTokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
import gc
from concurrent.futures import ThreadPoolExecutor
import threading

from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information and metadata"""
    name: str
    model_type: str
    max_length: int
    min_length: int
    is_loaded: bool = False
    load_time: Optional[float] = None
    memory_usage: Optional[int] = None

@dataclass
class SummaryResult:
    """Summary result with metadata"""
    text: str
    confidence: float
    model_used: str
    processing_time: float
    token_count: int

class ModelManager:
    """
    Production-grade model manager with:
    - Singleton pattern for memory efficiency
    - Async loading and inference
    - Model ensemble and voting
    - Automatic fallback
    - Memory management
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.device = "cuda" if torch.cuda.is_available() and settings.ENABLE_GPU else "cpu"
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    async def initialize(self):
        """Initialize and load models asynchronously"""
        logger.info("Initializing models...")
        
        # Define model configurations
        model_configs = {
            "facebook/bart-large-cnn": {
                "type": "bart",
                "max_length": 142,
                "min_length": 56,
                "priority": 1
            },
            "google/pegasus-cnn_dailymail": {
                "type": "pegasus", 
                "max_length": 128,
                "min_length": 32,
                "priority": 2
            },
            "t5-large": {
                "type": "t5",
                "max_length": 200,
                "min_length": 50,
                "priority": 3
            }
        }
        
        # Load models with priority order
        for model_name in settings.SUMMARIZATION_MODELS:
            if model_name in model_configs:
                config = model_configs[model_name]
                try:
                    await self._load_model(model_name, config)
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    continue
        
        if not self.models:
            raise RuntimeError("No models could be loaded")
        
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    async def _load_model(self, model_name: str, config: Dict[str, Any]):
        """Load a specific model asynchronously"""
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load in thread to avoid blocking
            model, tokenizer = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._load_model_sync, model_name, config
            )
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Store model info
            self.model_info[model_name] = ModelInfo(
                name=model_name,
                model_type=config["type"],
                max_length=config["max_length"],
                min_length=config["min_length"],
                is_loaded=True
            )
            
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            raise
    
    def _load_model_sync(self, model_name: str, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Synchronous model loading"""
        model_type = config["type"]
        
        if model_type == "bart":
            tokenizer = BartTokenizer.from_pretrained(
                model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
            model = BartForConditionalGeneration.from_pretrained(
                model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
        elif model_type == "pegasus":
            tokenizer = PegasusTokenizer.from_pretrained(
                model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
            model = PegasusForConditionalGeneration.from_pretrained(
                model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
        elif model_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(
                model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        return model, tokenizer
    
    async def summarize_text(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        use_ensemble: bool = True
    ) -> List[SummaryResult]:
        """
        Generate summaries using single model or ensemble
        """
        if not self.models:
            raise RuntimeError("No models available")
        
        if use_ensemble and len(self.models) > 1:
            return await self._ensemble_summarize(text, max_length, min_length)
        else:
            # Use best available model
            model_name = next(iter(self.models.keys()))
            result = await self._single_model_summarize(
                text, model_name, max_length, min_length
            )
            return [result]
    
    async def _ensemble_summarize(
        self, 
        text: str, 
        max_length: Optional[int], 
        min_length: Optional[int]
    ) -> List[SummaryResult]:
        """Generate summaries using model ensemble"""
        tasks = []
        
        for model_name in self.models.keys():
            task = self._single_model_summarize(
                text, model_name, max_length, min_length
            )
            tasks.append(task)
        
        # Run models in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_results = [
            r for r in results 
            if isinstance(r, SummaryResult)
        ]
        
        if not valid_results:
            raise RuntimeError("All models failed to generate summaries")
        
        return valid_results
    
    async def _single_model_summarize(
        self,
        text: str,
        model_name: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> SummaryResult:
        """Generate summary using a single model"""
        import time
        start_time = time.time()
        
        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            model_info = self.model_info[model_name]
            
            # Use model defaults if not specified
            max_len = max_length or model_info.max_length
            min_len = min_length or model_info.min_length
            
            # Run inference in thread
            summary_text = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_summary_sync,
                text, model, tokenizer, model_name, max_len, min_len
            )
            
            processing_time = time.time() - start_time
            
            # Calculate confidence (simplified)
            confidence = min(0.95, 0.7 + (len(summary_text) / max_len) * 0.25)
            
            return SummaryResult(
                text=summary_text,
                confidence=confidence,
                model_used=model_name,
                processing_time=processing_time,
                token_count=len(tokenizer.encode(summary_text))
            )
            
        except Exception as e:
            logger.error(f"Error in {model_name}: {e}")
            raise
    
    def _generate_summary_sync(
        self,
        text: str,
        model: Any,
        tokenizer: Any,
        model_name: str,
        max_length: int,
        min_length: int
    ) -> str:
        """Synchronous summary generation"""
        
        # Prepare input based on model type
        if "t5" in model_name.lower():
            input_text = f"summarize: {text}"
        else:
            input_text = text
        
        # Tokenize
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode
        summary = tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True
        )
        
        return summary.strip()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check model health status"""
        status = {
            "total_models": len(self.model_info),
            "loaded_models": len(self.models),
            "device": self.device,
            "models": {}
        }
        
        for name, info in self.model_info.items():
            status["models"][name] = {
                "loaded": info.is_loaded,
                "type": info.model_type,
                "max_length": info.max_length
            }
        
        return status
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up models...")
        
        # Clear models
        for model in self.models.values():
            del model
        
        self.models.clear()
        self.tokenizers.clear()
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.executor.shutdown(wait=True)
        logger.info("Model cleanup complete")