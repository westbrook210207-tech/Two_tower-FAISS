# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – ingest.py
#
# Run ONCE (or whenever you retrain your model) to:
#   a) Push movie metadata into PostgreSQL
#   b) Push precomputed movie embeddings into Qdrant
#   c) Push precomputed user embeddings into Qdrant
#
# This script mirrors exactly what your notebook already does, but writes the
# results to persistent stores instead of keeping them in memory.
# ─────────────────────────────────────────────────────────────────────────────

import asyncio

import asyncpg
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import get_settings

# ── Config ────────────────────────────────────────────────────────────────────
_cfg = get_settings()

POSTGRES_DSN = _cfg.postgres_dsn  # This line basically means Connect to the PostgreSQL (postgresql) server running on my computer (localhost), through port 5432, using username recsys (1st one), password recsys (2nd one), and connect to the recsys database (last one).

# Qdrant doesn't need a connection string like PG, just a host and port is enough and QdrantClient will handle the rest.
QDRANT_HOST = _cfg.qdrant_host  # The host where Qdrant is running
QDRANT_PORT = _cfg.qdrant_port  # The port where Qdrant is listening for connections

DATA_DIR      = "../ml-100k"   # path to the ml-100k folder
MODEL_PATH    = "two_tower.pt" # path to your saved model weights (see below)
EMBEDDING_DIM = 64
MOVIE_COLLECTION  = "movies"
USER_COLLECTION   = "users"

# ─────────────────────────────────────────────────────────────────────────────
# A.  Re-create the exact same feature matrices that the notebook built
#     (same code as cells 2-50 in your notebook)
# ─────────────────────────────────────────────────────────────────────────────

def build_features():
    # ── Load raw files ────────────────────────────────────────────────────────
    genre_cols = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    movie_cols = (
        ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL']
        + genre_cols
    )
    movie = pd.read_csv(
        f"{DATA_DIR}/u.item", sep="|", names=movie_cols,
        encoding="latin-1", engine="python"
    ).drop(columns=["video_release_date", "IMDb_URL"])
    movie["movie_id"] = movie["movie_id"] - 1          # 0-indexed

    user_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
    users = pd.read_csv(
        f"{DATA_DIR}/u.user", sep="|", names=user_cols, engine="python"
    )
    users["user_id"] = users["user_id"] - 1            # 0-indexed

    # ── Movie features (same as notebook cells 16-22) ─────────────────────────
    release_year = pd.to_numeric(
        movie["release_date"].str.split("-").str[-1], errors="coerce"
    )
    release_year = release_year.fillna(release_year.median())

    scaler = StandardScaler()
    movie["release_year_scaled"] = scaler.fit_transform(
        release_year.values.reshape(-1, 1)
    )

    genre_array = movie[genre_cols].values                           # (N, 19)
    year_array  = movie["release_year_scaled"].values.reshape(-1, 1) # (N, 1)
    genre_year  = genre_array * year_array                           # (N, 19)

    movie_features = np.hstack([genre_array, year_array, genre_year]).astype("float32")

    # ── User features (exactly mirrors notebook cells 26-40) ──────────────────
    # Step 1: encode gender
    users["gender_idx"] = users["gender"].map({"M": 0, "F": 1})
    users = users.drop(columns="gender")

    # Step 2: scale age
    age_scaler = StandardScaler()
    users["age"] = age_scaler.fit_transform(users[["age"]])

    # Step 3: one-hot occupation (21 categories in ml-100k → 21 cols)
    occ_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    occ_encoded = occ_encoder.fit_transform(users[["occupation"]])
    occ_cols = occ_encoder.get_feature_names_out(["occupation"])
    occ_df = pd.DataFrame(occ_encoded, columns=occ_cols, index=users.index)
    users = pd.concat([users, occ_df], axis=1).drop(columns="occupation")

    # Step 4: zip_region one-hot (first digit of zip, 'intl' for non-numeric)
    # This is exactly what notebook cells 35-36 do
    users["zip_region"] = users["zip_code"].astype(str).str[0]
    users["zip_region"] = users["zip_region"].where(
        users["zip_region"].str.isdigit(), "intl"
    )
    zip_onehot = pd.get_dummies(users["zip_region"], prefix="zip_region", dtype=int)
    users = users.drop(columns=["zip_code", "zip_region"])

    # Step 5: stack — same as notebook cell 40
    # user_feature_cols = users.columns[1:] (everything except user_id)
    user_feature_cols = users.columns[1:].tolist()
    user_features = np.hstack([
        users[user_feature_cols].values,   # age + gender_idx + 21 occ cols = 23
        zip_onehot.values,                 # 11 zip_region cols
    ]).astype("float32")
    # total: 23 + 11 = 34 columns — matches model checkpoint

    return movie, users, movie_features, user_features


# ─────────────────────────────────────────────────────────────────────────────
# B.  Re-create your TwoTower model architecture and load saved weights
# ─────────────────────────────────────────────────────────────────────────────

class TwoTower(nn.Module):
    def __init__(self, user_input_dim: int, movie_input_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.user_mlp = nn.Sequential(
            nn.Linear(user_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        self.movie_mlp = nn.Sequential(
            nn.Linear(movie_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, user_features, movie_features):
        user_emb  = self.user_mlp(user_features)
        movie_emb = self.movie_mlp(movie_features)
        return (user_emb * movie_emb).sum(dim=1)


def compute_embeddings(model_path, movie_features, user_features):
    """Load the trained model and run inference to get all embeddings."""
    movie_input_dim = movie_features.shape[1]
    user_input_dim  = user_features.shape[1]

    model = TwoTower(user_input_dim, movie_input_dim, EMBEDDING_DIM)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        movie_emb_np = (
            model.movie_mlp(torch.tensor(movie_features))
            .numpy()
            .astype("float32")
        )
        user_emb_np = (
            model.user_mlp(torch.tensor(user_features))
            .numpy()
            .astype("float32")
        )

    return movie_emb_np, user_emb_np


# ─────────────────────────────────────────────────────────────────────────────
# C.  Push movie metadata → PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

async def push_postgres(movie: pd.DataFrame):
    """
    Create a `movies` table and insert every row from the ml-100k u.item file.
    Columns stored: movie_id (PK), title, release_date, and all 19 genre flags.
    """
    # We use "async with ... as conn" instead of plain "conn = await asyncpg.connect()".
    #
    # Why? Because "async with" is a context manager — it guarantees that the
    # connection is ALWAYS closed when the block exits, whether the code:
    #   a) finishes successfully, OR
    #   b) crashes halfway through with an exception
    #
    # Without it (the old way):
    #   conn = await asyncpg.connect(...)
    #   ... if something crashes here ...
    #   await conn.close()   ← this line is never reached → connection leak
    #
    # With context manager:
    #   async with await asyncpg.connect(...) as conn:
    #       ... even if this crashes ...
    #   ← conn.close() is called automatically by Python here, always
    #
    # For a one-shot script this is low risk, but it's the correct habit to build.
    async with await asyncpg.connect(POSTGRES_DSN) as conn: # Connect Python to PostgreSQL using the DSN defined above. This allows us to execute SQL commands against the database.

        # Create table (idempotent), this is stored exactly like how the u.item file is structured, with movie_id as the primary key and all genre flags as smallints (0 or 1).
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                movie_id      INTEGER PRIMARY KEY,
                title         TEXT,
                release_date  TEXT,
                unknown       SMALLINT,
                action        SMALLINT,
                adventure     SMALLINT,
                animation     SMALLINT,
                childrens     SMALLINT,
                comedy        SMALLINT,
                crime         SMALLINT,
                documentary   SMALLINT,
                drama         SMALLINT,
                fantasy       SMALLINT,
                film_noir     SMALLINT,
                horror        SMALLINT,
                musical       SMALLINT,
                mystery       SMALLINT,
                romance       SMALLINT,
                sci_fi        SMALLINT,
                thriller      SMALLINT,
                war           SMALLINT,
                western       SMALLINT
            )
        """)

        # Truncate and re-insert (idempotent re-run)
        await conn.execute("TRUNCATE movies") # execute is just a way that python let us do SQL commands

        genre_raw = [
            'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        rows = []
        for _, row in movie.iterrows(): # This code go through each row in the movie df and extract the row data and index. Since we don't need the index, we use _ to ignore it. Then we can access the row data using row[column_name].
            release = row.get("release_date", "") # check if release_date col exists in the df, if it does, get the value, if not, return an empty string.

            # This code basically means If release_date isn't a string, replace it with an empty string. Since a lot of the release_date values are NaN, we need to handle that case.
            if not isinstance(release, str): 
                release = "" # NaN becomes an empty string so asyncpg doesn't fail on TEXT columns
            rows.append((
                int(row["movie_id"]),
                row["movie_title"],
                release, # since we already handled NaN above, we can just use the release variable here instead of row["release_date"]
                *(int(row[g]) for g in genre_raw), # we unpack the genre Boolean columns into the tuple using * and a generator expression. 
            ))

        # executemany is a way to execute a SQL command for multiple rows at once, which is more efficient than doing it one by one with a for loop
        await conn.executemany("""
            INSERT INTO movies VALUES ($1,$2,$3,
              $4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22)
        """, rows) # $1 through $22 are placeholders for the values in each row, which correspond to the columns in the movies table (title, release_date, and 19 genres). This allows us to insert all the movie data into the database in one go.

    # conn.close() is NOT needed here — the "async with" block above handles it automatically.
    print(f"✅  PostgreSQL: inserted {len(rows)} movies into `movies` table.")


# ─────────────────────────────────────────────────────────────────────────────
# D.  Push embeddings → Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def push_qdrant(movie_emb_np: np.ndarray, user_emb_np: np.ndarray):
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    for collection_name, embeddings in [
        (MOVIE_COLLECTION, movie_emb_np),
        (USER_COLLECTION,  user_emb_np),
    ]:
        # Recreate collection (drop + create for idempotent re-runs)
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embeddings.shape[1],
                distance=Distance.DOT,   # we'll use dot product to compute the score between user and movie embeddings.
            ),
        )

        client.upload_points(
            collection_name=collection_name,
            points=[
                PointStruct(id=i, vector=embeddings[i].tolist()) # PointStruct packages the embedding vector with its ID for Qdrant
                for i in range(len(embeddings))
            ],
            wait=True,
        )
        print(f"✅  Qdrant '{collection_name}': uploaded {len(embeddings)} vectors.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print("🔄  Building features …")
    movie, users, movie_features, user_features = build_features()

    print("🔄  Computing embeddings from saved model …")
    movie_emb_np, user_emb_np = compute_embeddings(MODEL_PATH, movie_features, user_features)

    print("🔄  Pushing movie metadata → PostgreSQL …")
    await push_postgres(movie)

    print("🔄  Pushing embeddings → Qdrant …")
    push_qdrant(movie_emb_np, user_emb_np)

    print("\n🎉  Ingestion complete! You can now start the FastAPI server.")


if __name__ == "__main__":
    asyncio.run(main())

