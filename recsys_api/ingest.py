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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import asyncio
import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────────────────────────────
POSTGRES_DSN = "postgresql://recsys:recsys@localhost:5432/recsys"
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333

DATA_DIR     = "../ml-100k"       # path to the ml-100k folder
MODEL_PATH   = "two_tower.pt"    # path to your saved model weights (see below)
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
    from sklearn.preprocessing import OneHotEncoder
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
    conn = await asyncpg.connect(POSTGRES_DSN)

    # Create table (idempotent)
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
    await conn.execute("TRUNCATE movies")

    genre_raw = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    rows = []
    for _, row in movie.iterrows():
        release = row.get("release_date", "")
        # NaN becomes an empty string so asyncpg doesn't fail on TEXT columns
        if not isinstance(release, str):
            release = ""
        rows.append((
            int(row["movie_id"]),
            row["movie_title"],
            release,
            *(int(row[g]) for g in genre_raw),
        ))

    await conn.executemany("""
        INSERT INTO movies VALUES ($1,$2,$3,
          $4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22)
    """, rows)

    await conn.close()
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
                distance=Distance.DOT,   # same as your notebook (dot product)
            ),
        )

        client.upload_points(
            collection_name=collection_name,
            points=[
                PointStruct(id=i, vector=embeddings[i].tolist())
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

