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

from config import get_settings

# ── Connection settings (loaded once from .env via pydantic-settings) ─────────
_cfg = get_settings()

QDRANT_HOST  = _cfg.qdrant_host
QDRANT_PORT  = _cfg.qdrant_port
POSTGRES_DSN = _cfg.postgres_dsn

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
        self._client: AsyncQdrantClient | None = None # When we create qdrant_store = QdrantStore(), it creates qdrant_store._client = None because we haven't connected yet. 
        
    #Then when we call qdrant_store.connect(), it creates the AsyncQdrantClient and assigns it to self._client. This way, we can use self._client in other methods like get_user_vector() and search_movies() without having to create a new client each time.
    async def connect(self):
        self._client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT) # Now qdrant_store.client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT). This basically means "Create an object capable of communicating with the Qdrant server at localhost:6333."

    async def close(self):
        if self._client: # if self._client is connected, then we can close it after we're done with it. But if self._client is None, then we don't need to do anything because there's nothing to close. This is a safety check to avoid errors when closing the client.
            await self._client.close()

    async def get_user_vector(self, user_id: int) -> List[float]: # we can see the parameter the function takes in is a user ID, the response model is a list of floats, which is the user embeddings.
        """
        Retrieve the precomputed user embedding from the 'users' collection.
        Point IDs in Qdrant map directly to user_id (0-indexed, same as notebook).
        """
        result = await self._client.retrieve(
            collection_name=USER_COLLECTION,
            ids=[user_id], # We have user_id as a list since Qdrant can retrieve multiple points at once, but we only want one user embedding here.
            with_vectors=True, # this basically means "Don't just give me information about the point. Give me the actual embedding vector too."
        )
        if not result:
            raise ValueError(f"User {user_id} not found in Qdrant.")
        return result[0].vector  # return the user embedding as a list[float]

    async def search_movies(
        self, user_vector: List[float], top_k: int
    ) -> List[int]: # take in the user_vector from above and top_k, return a list[int] of movie IDs (0-indexed) sorted by score descending.
        """
        Run an ANN (dot-product) search over the 'movies' collection.
        Returns a list of movie_ids sorted by score descending.

        This is equivalent to the notebook's client.query_points(...) call.
        """
        results = await self._client.query_points(
            collection_name=MOVIE_COLLECTION,
            query=user_vector,      # query based on the user embedding we just retrieved from above
            limit=top_k,
            with_payload=False, 
        )
        # results.points is a list of ScoredPoint; .id is the movie_id (0-indexed)
        return [int(r.id) for r in results.points] # Takes only the id, ignore other info like score, since we only need the movie IDs to fetch metadata from PG. The int() is needed because Qdrant returns the ID as a string, but we want it as an int for our database query.


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
        self._pool: asyncpg.Pool | None = None # Here we're storing a pool instead of a single PG connection. A connection pool is basically a collection of reusable DB connections.
                                                # This matters because your FastAPI application can receive many requests. You don't want every request to create a completely new PostgreSQL connection.

    async def connect(self):
        self._pool = await asyncpg.create_pool(POSTGRES_DSN) # We create a connection pool to the PostgreSQL database using the connection string defined in POSTGRES_DSN. 

    async def close(self):
        if self._pool: # if self._pool is connected, then we can close it after we're done with it. But if self._pool is None, then we don't need to do anything because there's nothing to close. This is a safety check to avoid errors when closing the pool.
            await self._pool.close()

    async def fetch_movies(self, movie_ids: List[int]) -> List[Dict[str, Any]]: # Take in a list of movie IDs from Qdrant, return a list of dict of string key such as ID, title, release_date with Any type values such as 12, "Toy Story", 2020-01-01. The order of the returned list matches the order of movie_ids.
        """
        Given a list of movie_ids (0-indexed), fetch their metadata from PG.
        Returns rows in the SAME ORDER as movie_ids so the caller can zip
        scores with metadata.
        """
        if not movie_ids:
            return []

        async with self._pool.acquire() as conn: # acquire means "use an available connection from the pool". This is important because we don't want to create a new connection for every request. Instead, we reuse existing connections from the pool.
            rows = await conn.fetch(
                """
                SELECT movie_id, title, release_date,
                       action, adventure, animation, childrens, comedy,
                       crime, documentary, drama, fantasy, film_noir,
                       horror, musical, mystery, romance, sci_fi,
                       thriller, war, western
                FROM   movies
                WHERE  movie_id = ANY($1::int[]) -- Takes in a list of movie_ids as int and returns all rows that match any of the IDs in the list. The $1 is a placeholder for the first argument passed to conn.fetch(), which is movie_ids. The ::int[] tells PostgreSQL that we're passing an array of integers.
                """,
                movie_ids, # This is the argument that gets passed to the SQL query above. It replaces $1 in the query with the list of movie_ids we want to fetch from the database.
            ) # Then when the "async with" block ends, the connection is returned to the pool.

        
        row_by_id = {r["movie_id"]: dict(r) for r in rows} # This line creates a dictionary where the keys are movie IDs and the values are dictionaries of the corresponding row data. This allows for quick lookups of movie metadata by ID. 
        return [row_by_id[mid] for mid in movie_ids if mid in row_by_id] # asyncpg returns rows in whatever order PG prefers so we need to re-sort based on Qdrant ranking order.


# ─────────────────────────────────────────────────────────────────────────────
# Singletons – created once, shared across the whole app lifetime
# ─────────────────────────────────────────────────────────────────────────────

qdrant_store   = QdrantStore()
postgres_store = PostgresStore()

