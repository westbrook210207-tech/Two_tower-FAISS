# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – main.py
#
# FastAPI application.
#
# One endpoint:   GET /recommend/{user_id}?top_k=10
#
# Full pipeline per request:
#   1. Validate user_id
#   2. Fetch user embedding from Qdrant  (O(1) point lookup)
#   3. ANN search: top-K movie IDs from Qdrant  (fast vector search)
#   4. Fetch movie metadata from PostgreSQL  (indexed point-lookup by PK)
#   5. Return structured JSON response
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from db import qdrant_store, postgres_store


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager # this goes default with the lifespan
async def lifespan(app: FastAPI):
    """
    FastAPI 'lifespan' hook:
      - Opens DB connections on startup
      - Closes them gracefully on shutdown

    This runs ONCE when the server starts, NOT on every request.
    """
    print("🚀 Connecting to Qdrant and PostgreSQL …")
    await qdrant_store.connect()    # creates AsyncQdrantClient
    await postgres_store.connect()  # creates asyncpg connection pool
    print("✅ Ready to serve requests.")

    yield  # ← server runs here and wait until shutdown is triggered (e.g. Ctrl+C)

    print("🛑 Shutting down, closing connections …")
    await qdrant_store.close()
    await postgres_store.close()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RecSys API – Two-Tower CBF",
    description="Content-based Two-Tower model served via Qdrant + PostgreSQL",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS so web browsers (and Vercel) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────────────────────────────────────

class MovieRecommendation(BaseModel):
    movie_id:     int
    title:        str
    release_date: str
    genres:       List[str]   # human-readable list, e.g. ["Action", "Drama"]


class RecommendResponse(BaseModel):
    user_id:      int
    top_k:        int
    recommended:  List[MovieRecommendation]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: turn DB row genre flags into a list of genre strings
# ─────────────────────────────────────────────────────────────────────────────

GENRE_COLUMNS = [
    "unknown",
    "action", "adventure", "animation", "childrens", "comedy",
    "crime", "documentary", "drama", "fantasy", "film_noir",
    "horror", "musical", "mystery", "romance", "sci_fi",
    "thriller", "war", "western",
]

# Since Postgres stores genres as a set of boolean columns, we need to read that boolean data and convert it back to the original list of genres. For example, if a movie has action=1, comedy=1 and animation=0, we want to return ["Action", "Comedy"].
def extract_genres(row: dict) -> List[str]:
    return [col.replace("_", "-").title() for col in GENRE_COLUMNS if row.get(col)] # This one helps transforming "sci_fi" into "Sci-Fi" and "film_noir" into "Film-Noir". The .title() method capitalizes the first letter of each word, and the .replace("_", "-") replaces underscores with hyphens. The if row.get(col) part filters out any genres that are not present (i.e., have a value of 0 or False in the database row).


# ─────────────────────────────────────────────────────────────────────────────
# THE endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    summary="Get top-K movie recommendations for a user",
)
async def recommend(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=500, description="Number of recommendations"),
):
    """
    Full pipeline:

    ```
    Client  →  FastAPI  →  Qdrant (get user vector)
                        →  Qdrant (ANN search top-K movies)
                        →  PostgreSQL (fetch movie metadata)
                        →  Client
    ```
    """

    # ── Step 1: Validate user_id ──────────────────────────────────────────────
    # ml-100k has 943 users (0-indexed: 0 … 942)
    if user_id < 0 or user_id > 942:
        raise HTTPException(
            status_code=400,
            detail=f"user_id must be between 0 and 942. Got {user_id}.",
        )

    # ── Step 2: Fetch user embedding from Qdrant ──────────────────────────────
    # The 'users' collection stores one vector per user_id.
    # This is a direct point-retrieval by ID — O(1), not a search.
    try:
        user_vector = await qdrant_store.get_user_vector(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # ── Step 3: ANN search – find top_k closest movie embeddings ─────────────
    # Qdrant computes dot-product(user_vector, movie_vector) for all movies
    # and returns the top_k highest-scoring movie IDs.
    # This is the "retrieval" stage of the recsys pipeline.
    movie_ids = await qdrant_store.search_movies(user_vector, top_k)

    # ── Step 4: Fetch movie metadata from PostgreSQL ──────────────────────────
    # We know WHICH movies to show; now we need WHAT to show (title, year, genres).
    # A single SQL query fetches all rows by primary key – very fast.
    movies_data = await postgres_store.fetch_movies(movie_ids)

    # ── Step 5: Build the response ────────────────────────────────────────────
    # recommendations is List[MovieRecommendation]
    recommendations = [
        MovieRecommendation( # We call out Movierecommendation model here for validation and type checking. This ensures that the data we return matches the expected schema.
            movie_id     = row["movie_id"],
            title        = row["title"],
            release_date = row.get("release_date", ""),
            genres       = extract_genres(row),
        )
        for row in movies_data
    ]

    return RecommendResponse( # We call out RecommendResponse model here for validation and type checking. This ensures that the data we return matches the expected schema.
        user_id=user_id,
        top_k=top_k,
        recommended=recommendations, # Plug recommendations into the response model.
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}

