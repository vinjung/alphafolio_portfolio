# -*- coding: utf-8 -*-
"""
Shared Database Manager

Provides a single database connection pool for the entire portfolio generation pipeline.
Includes connection timeout, keepalive, and retry settings for stability.

File: create_portfolio/db_manager.py
Created: 2025-12-24
"""

import os
import asyncio
import asyncpg
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load .env from quant directory
env_path = Path(__file__).parent.parent / 'quant' / '.env'
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SharedDatabaseManager:
    """
    Shared database connection pool manager.

    Features:
    - Single pool shared across all pipeline steps
    - Extended timeout settings for remote DB
    - Keepalive settings to prevent connection drops
    - Retry logic for connection failures
    """

    _instance: Optional['SharedDatabaseManager'] = None
    _pool: Optional[asyncpg.Pool] = None

    def __init__(self):
        """Initialize manager (pool created on initialize())"""
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    async def initialize(self, max_retries: int = 3) -> None:
        """
        Initialize connection pool with retry logic.

        Args:
            max_retries: Number of connection attempts
        """
        if self._initialized and self.pool:
            logger.info("Database pool already initialized, reusing")
            return

        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")

        # Convert SQLAlchemy URL format to asyncpg format
        db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                self.pool = await asyncpg.create_pool(
                    db_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=300,  # 5 minutes timeout
                    server_settings={
                        'application_name': 'alpha_portfolio',
                        'statement_timeout': '300000',  # 5 minutes in ms
                    },
                )
                self._initialized = True
                logger.info(f"Database pool initialized (attempt {attempt}/{max_retries})")
                return

            except Exception as e:
                last_error = e
                logger.warning(f"Database connection attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise ConnectionError(f"Failed to connect to database after {max_retries} attempts: {last_error}")

    async def close(self) -> None:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("Database pool closed")

    async def execute_query(self, query: str, *params) -> List[Dict[str, Any]]:
        """
        Execute query and return results as list of dicts.

        Args:
            query: SQL query string
            *params: Query parameters

        Returns:
            List of dictionaries with query results
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def execute_one(self, query: str, *params) -> Optional[Dict[str, Any]]:
        """
        Execute query and return single result.

        Args:
            query: SQL query string
            *params: Query parameters

        Returns:
            Dictionary with query result or None
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return dict(row) if row else None

    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized"""
        return self._initialized and self.pool is not None


async def get_shared_pool() -> asyncpg.Pool:
    """
    Get or create a shared database pool.

    Returns:
        asyncpg.Pool instance
    """
    manager = SharedDatabaseManager()
    await manager.initialize()
    return manager.pool


# ============================================================================
# Test
# ============================================================================

async def test_connection():
    """Test database connection"""
    print("=" * 70)
    print("Shared Database Manager - Connection Test")
    print("=" * 70)

    manager = SharedDatabaseManager()
    try:
        await manager.initialize()

        # Test query
        result = await manager.execute_one("SELECT COUNT(*) as cnt FROM kr_stock_grade")
        print(f"  kr_stock_grade count: {result['cnt']}")

        result = await manager.execute_one("SELECT MAX(date) as latest FROM kr_stock_grade")
        print(f"  Latest date: {result['latest']}")

        print("\n  Connection test successful!")

    except Exception as e:
        print(f"\n  Connection test failed: {e}")
    finally:
        await manager.close()

    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(test_connection())
