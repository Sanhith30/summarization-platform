"""
Rate Limiting System
Redis-based rate limiting with sliding window approach
"""

import asyncio
import logging
import time
from typing import Dict, Optional
import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Production-grade rate limiter with:
    - Sliding window algorithm
    - Per-IP and global limits
    - Redis-based storage
    - Graceful fallback
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, list] = {}  # Fallback for when Redis is unavailable
        self.is_redis_available = False
    
    async def initialize(self):
        """Initialize Redis connection for rate limiting"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            await self.redis_client.ping()
            self.is_redis_available = True
            logger.info("Rate limiter initialized with Redis")
            
        except Exception as e:
            logger.warning(f"Redis unavailable for rate limiting: {e}. Using local fallback.")
            self.is_redis_available = False
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int = 60
    ) -> bool:
        """
        Check if request is within rate limit
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if request is allowed, False if rate limited
        """
        
        if self.is_redis_available and self.redis_client:
            return await self._check_redis_rate_limit(identifier, limit, window_seconds)
        else:
            return await self._check_local_rate_limit(identifier, limit, window_seconds)
    
    async def _check_redis_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int
    ) -> bool:
        """Redis-based sliding window rate limiting"""
        
        try:
            current_time = time.time()
            window_start = current_time - window_seconds
            
            key = f"rate_limit:{identifier}"
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window_seconds + 1)
            
            results = await pipe.execute()
            
            current_count = results[1]  # Result of zcard
            
            return current_count < limit
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to local check
            return await self._check_local_rate_limit(identifier, limit, window_seconds)
    
    async def _check_local_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int
    ) -> bool:
        """Local memory-based rate limiting (fallback)"""
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Initialize if not exists
        if identifier not in self.local_cache:
            self.local_cache[identifier] = []
        
        # Remove old entries
        self.local_cache[identifier] = [
            timestamp for timestamp in self.local_cache[identifier]
            if timestamp > window_start
        ]
        
        # Check limit
        if len(self.local_cache[identifier]) >= limit:
            return False
        
        # Add current request
        self.local_cache[identifier].append(current_time)
        
        return True
    
    async def get_rate_limit_info(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int = 60
    ) -> Dict[str, int]:
        """Get current rate limit status"""
        
        try:
            if self.is_redis_available and self.redis_client:
                current_time = time.time()
                window_start = current_time - window_seconds
                key = f"rate_limit:{identifier}"
                
                # Count requests in current window
                current_count = await self.redis_client.zcount(key, window_start, current_time)
                
                return {
                    "current_requests": current_count,
                    "limit": limit,
                    "remaining": max(0, limit - current_count),
                    "reset_time": int(current_time + window_seconds)
                }
            
            else:
                # Local fallback
                current_time = time.time()
                window_start = current_time - window_seconds
                
                if identifier not in self.local_cache:
                    current_count = 0
                else:
                    current_count = len([
                        t for t in self.local_cache[identifier]
                        if t > window_start
                    ])
                
                return {
                    "current_requests": current_count,
                    "limit": limit,
                    "remaining": max(0, limit - current_count),
                    "reset_time": int(current_time + window_seconds)
                }
                
        except Exception as e:
            logger.error(f"Rate limit info failed: {e}")
            return {
                "current_requests": 0,
                "limit": limit,
                "remaining": limit,
                "reset_time": int(time.time() + window_seconds)
            }
    
    async def cleanup(self):
        """Cleanup rate limiter resources"""
        if self.redis_client:
            await self.redis_client.close()
        
        self.local_cache.clear()
        logger.info("Rate limiter cleanup complete")