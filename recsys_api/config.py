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

# Since this is a local production, we'll have default value revealed. 
# But in real production, these values are required to be manually set

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
    @property # This line basically says "Treat this method like an attribute instead of a method."
    def postgres_dsn(self) -> str: # So we can retrieve postgres dsn by calling _cfg.postgres_dsn instead of _cfg.postgres_dsn()
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}" # Python has a special rule: adjacent string literals are automatically concatenated. We split the out for readability
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}" # So this is not 2 strings but it's actually f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    # Why do we even want _cfg.postgres_dsn instead of _cfg.postgres_dsn()
    # postgres_dsn is actually a function combining the 5 env var of PG into a string (it's doing something so it's a method), but since we wanna call out PG DSN as an object and not like tell it to do something, that's explain the self.postgres_dsn fits better than self.postgres_dsn()

    # Tell pydantic-settings where to find the .env file
    model_config = SettingsConfigDict(
        env_file=".env",          # reads .env from the current working directory
        env_file_encoding="utf-8", # Tell pydantic to interpret the .env file as UTF-8 text.
        case_sensitive=False,     # POSTGRES_HOST and postgres_host both match
    )


@lru_cache  # Called once per process; the same Settings object is reused forever after
def get_settings() -> Settings:
    return Settings()

