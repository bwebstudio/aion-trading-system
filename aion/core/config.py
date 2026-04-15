"""
aion.core.config
────────────────
System configuration loaded from environment variables / .env file.

Usage:
    from aion.core.config import get_config
    cfg = get_config()

Rules:
- All settings have safe defaults for local development.
- Sensitive values (MT5 credentials) are never logged.
- Call cfg.ensure_dirs() once at startup to create the data directory tree.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aion.core.enums import SystemEnvironment, SystemMode


class AionConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AION_",
        case_sensitive=False,
        extra="ignore",
    )

    # ── System ────────────────────────────────
    environment: SystemEnvironment = SystemEnvironment.RESEARCH
    mode: SystemMode = SystemMode.PAPER
    log_level: str = "INFO"

    # ── Timezone ──────────────────────────────
    # Operator's local timezone — used for display and debug logs only.
    # Does NOT affect any market calculations.
    local_timezone: str = "Europe/Madrid"

    # ── Data directories ──────────────────────
    data_root: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    normalized_data_dir: Path = Path("data/normalized")
    features_data_dir: Path = Path("data/features")
    snapshots_data_dir: Path = Path("data/snapshots")
    samples_data_dir: Path = Path("data/samples")

    # ── Pipeline defaults ──────────────────────
    default_symbol: str = "EURUSD"

    # ── MetaTrader5 (live/historical only) ────
    mt5_login: int | None = Field(default=None)
    mt5_password: str | None = Field(default=None)
    mt5_server: str | None = Field(default=None)

    # ── Validators ────────────────────────────

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    # ── Helpers ───────────────────────────────

    @property
    def data_dirs(self) -> list[Path]:
        return [
            self.raw_data_dir,
            self.normalized_data_dir,
            self.features_data_dir,
            self.snapshots_data_dir,
            self.samples_data_dir,
        ]

    def ensure_dirs(self) -> None:
        """Create all data directories if they do not exist."""
        for directory in self.data_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def raw_dir_for(self, symbol: str) -> Path:
        path = self.raw_data_dir / symbol.upper()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def features_dir_for(self, symbol: str) -> Path:
        path = self.features_data_dir / symbol.upper()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def snapshots_dir_for(self, symbol: str) -> Path:
        path = self.snapshots_data_dir / symbol.upper()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def is_production(self) -> bool:
        return self.environment == SystemEnvironment.PRODUCTION

    def is_live(self) -> bool:
        return self.mode == SystemMode.LIVE

    def __repr__(self) -> str:
        return (
            f"AionConfig(environment={self.environment}, mode={self.mode}, "
            f"symbol={self.default_symbol}, tz={self.local_timezone})"
        )


@lru_cache(maxsize=1)
def get_config() -> AionConfig:
    """
    Return the singleton config instance.

    Cached after first call.  In tests, call get_config.cache_clear()
    before each test that changes environment variables.
    """
    return AionConfig()
