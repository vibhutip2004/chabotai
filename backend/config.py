"""Backend configuration - all values read from environment variables only.

Never hardcode secrets here. Copy .env.example to .env and fill in the values.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


def _require(name: str) -> str:
    """Read a required env var; raise on blank/missing."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is not set. "
            "Copy .env.example to .env and fill in the value."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# Required
SECRET_KEY: str = _require("SECRET_KEY")

# Optional LLM keys
GROQ_API_KEY: str = _optional("GROQ_API_KEY")
OPENAI_API_KEY: str = _optional("OPENAI_API_KEY")
LLM_PROVIDER: str = _optional("LLM_PROVIDER", "groq")

# Database
DATABASE_URL: str = _optional("DATABASE_URL", "sqlite:///./docuquery.db")
if DATABASE_URL.startswith("sqlite:///"):
    _raw = DATABASE_URL[len("sqlite:///"):]
    if _raw.startswith("/"):
        DB_PATH: str = _raw
    else:
        DB_PATH = str(BASE_DIR / _raw.lstrip("./"))
else:
    DB_PATH = DATABASE_URL

# Token settings
TOKEN_EXPIRE_MINUTES: int = int(_optional("TOKEN_EXPIRE_MINUTES", "60"))

# File uploads
UPLOAD_DIR: str = _optional("UPLOAD_DIR", str(BASE_DIR / "uploads"))
