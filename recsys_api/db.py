# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – db.py
#
# Two thin async clients that the API will use at request time:
#   • QdrantStore  → ANN search for top-K movie IDs given a user embedding
#   • PostgresStore → fetch movie metadata (title, year, genres) by IDs
#
# Both are initialised once at startup and shared across all requests via
# FastAPI's dependency-injection system.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Dict, Any

import asyncpg
from qdrant_client import AsyncQdrantClient


QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333
POSTGRES_DSN = "postgresql://recsys:recsys@localhost:5432/recsys"

MOVIE_COLLECTION = "movies"
USER_COLLECTION  = "users"


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant helpers
# ─────────────────────────────────────────────────────────────────────────────

class QdrantStore:
    """
    Wraps the async Qdrant client.

    Two operations:
      1. get_user_vector(user_id)        → fetch stored user embedding
      2. search_movies(vector, top_k)    → ANN search over the movie collection
    """

    def __init__(self):
        self._client: AsyncQdrantClient | None = None

    async def connect(self):
        self._client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    async def close(self):
        if self._client:
            await self._client.close()

    async def get_user_vector(self, user_id: int) -> List[float]:
        """
        Retrieve the precomputed user embedding from the 'users' collection.
        Point IDs in Qdrant map directly to user_id (0-indexed, same as notebook).
        """
        result = await self._client.retrieve(
            collection_name=USER_COLLECTION,
            ids=[user_id],
            with_vectors=True,
        )
        if not result:
            raise ValueError(f"User {user_id} not found in Qdrant.")
        return result[0].vector  # List[float] of length embedding_dim

    async def search_movies(
        self, user_vector: List[float], top_k: int
    ) -> List[int]:
        """
        Run an ANN (dot-product) search over the 'movies' collection.
        Returns a list of movie_ids sorted by score descending.

        This is equivalent to the notebook's client.query_points(...) call.
        """
        results = await self._client.query_points(
            collection_name=MOVIE_COLLECTION,
            query=user_vector,      # the 64-dim user embedding vector
            limit=top_k,
            with_payload=False,
        )
        # results.points is a list of ScoredPoint; .id is the movie_id (0-indexed)
        return [int(r.id) for r in results.points]


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL helpers
# ─────────────────────────────────────────────────────────────────────────────

class PostgresStore:
    """
    Wraps an asyncpg connection pool.

    One operation:
      fetch_movies(movie_ids) → list of dicts with title, release_date, genres
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None

    async def connect(self):
        self._pool = await asyncpg.create_pool(POSTGRES_DSN)

    async def close(self):
        if self._pool:
            await self._pool.close()

    async def fetch_movies(self, movie_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Given a list of movie_ids (0-indexed), fetch their metadata from PG.
        Returns rows in the SAME ORDER as movie_ids so the caller can zip
        scores with metadata.
        """
        if not movie_ids:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT movie_id, title, release_date,
                       action, adventure, animation, childrens, comedy,
                       crime, documentary, drama, fantasy, film_noir,
                       horror, musical, mystery, romance, sci_fi,
                       thriller, war, western
                FROM   movies
                WHERE  movie_id = ANY($1::int[])
                """,
                movie_ids,
            )

        # asyncpg returns rows in whatever order PG prefers — re-sort to match
        row_by_id = {r["movie_id"]: dict(r) for r in rows}
        return [row_by_id[mid] for mid in movie_ids if mid in row_by_id]


# ─────────────────────────────────────────────────────────────────────────────
# Singletons – created once, shared across the whole app lifetime
# ─────────────────────────────────────────────────────────────────────────────

qdrant_store   = QdrantStore()
postgres_store = PostgresStore()

