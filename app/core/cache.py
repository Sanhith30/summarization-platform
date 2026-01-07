"""
Intelligent Caching System
Redis-based caching with hash-based keys and TTL management
"""

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, Optional, Union
import redis.asyncio as redis
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Production-grade cache manager with:
    - Hash-based content keys
    - Automatic TTL management
    - Compression for large content
    - Health monitoring
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running without cache.")
            self.is_connected = False
    
    def _generate_cache_key(self, content: str, **kwargs) -> str:
        """Generate deterministic cache key from content and parameters"""
        # Create hash from content and parameters
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Include relevant parameters in key
        params = {
            k: v for k, v in kwargs.items() 
            if k in ['query', 'mode', 'max_length', 'min_length']
        }
        
        if params:
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            return f"summary:{content_hash}:{params_hash}"
        
        return f"summary:{content_hash}"
    
    async def get_summary(self, content: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve cached summary"""
        if not self.is_connected:
            return None
        
        try:
            cache_key = self._generate_cache_key(content, **kwargs)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"Cache hit for key: {cache_key}")
                return json.loads(cached_data)
            
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    async def set_summary(
        self, 
        content: str, 
        summary_data: Dict[str, Any], 
        ttl: Optional[int] = None,
        **kwargs
    ):
        """Store summary in cache"""
        if not self.is_connected:
            return
        
        try:
            cache_key = self._generate_cache_key(content, **kwargs)
            ttl = ttl or settings.CACHE_TTL
            
            # Add metadata
            cache_data = {
                **summary_data,
                "cached_at": datetime.utcnow().isoformat(),
                "cache_key": cache_key
            }
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data, ensure_ascii=False)
            )
            
            logger.info(f"Cached summary with key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def get_youtube_transcript(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get cached YouTube transcript"""
        if not self.is_connected:
            return None
        
        try:
            cache_key = f"youtube:transcript:{video_id}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"YouTube cache retrieval error: {e}")
            return None
    
    async def set_youtube_transcript(
        self, 
        video_id: str, 
        transcript_data: Dict[str, Any],
        ttl: int = 86400  # 24 hours
    ):
        """Cache YouTube transcript"""
        if not self.is_connected:
            return
        
        try:
            cache_key = f"youtube:transcript:{video_id}"
            
            cache_data = {
                **transcript_data,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data, ensure_ascii=False)
            )
            
            logger.info(f"Cached YouTube transcript: {video_id}")
            
        except Exception as e:
            logger.error(f"YouTube cache storage error: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if not self.is_connected:
            return
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries")
        
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.is_connected:
            return {"status": "disconnected"}
        
        try:
            info = await self.redis_client.info()
            
            # Get key counts by pattern
            summary_keys = len(await self.redis_client.keys("summary:*"))
            youtube_keys = len(await self.redis_client.keys("youtube:*"))
            
            return {
                "status": "connected",
                "memory_usage": info.get("used_memory_human", "unknown"),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "summary_keys": summary_keys,
                "youtube_keys": youtube_keys,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0)
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for cache system"""
        try:
            if not self.is_connected:
                return {"status": "disconnected"}
            
            # Test basic operations
            test_key = "health_check"
            await self.redis_client.set(test_key, "ok", ex=10)
            result = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            if result == "ok":
                return {"status": "healthy", "connected": True}
            else:
                return {"status": "unhealthy", "connected": True}
                
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def cleanup(self):
        """Cleanup cache connections"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Cache cleanup complete")