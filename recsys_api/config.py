# ─────────────────────────────────────────────────────────────────────────────
# config.py
#
# Centralised settings via pydantic-settings.
# All values are read from .env (or from the real environment if already set,
# e.g. when injected by Docker Compose).
#
# Usage anywhere in the project:
#   from config import get_settings
#   cfg = get_settings()
#   print(cfg.postgres_dsn)
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    postgres_user:     str = "recsys"
    postgres_password: str = "recsys"
    postgres_host:     str = "localhost"
    postgres_port:     int = 5432
    postgres_db:       str = "recsys"

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ── FastAPI ────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ── Derived DSN (computed property, not an env var) ────────────────────────
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Tell pydantic-settings where to find the .env file
    model_config = SettingsConfigDict(
        env_file=".env",          # reads .env from the current working directory
        env_file_encoding="utf-8",
        case_sensitive=False,     # POSTGRES_HOST and postgres_host both match
    )


@lru_cache  # Called once per process; the same Settings object is reused forever after
def get_settings() -> Settings:
    return Settings()

