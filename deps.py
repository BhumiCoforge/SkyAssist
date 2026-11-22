import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from config import settings

_admin_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def require_admin(key: str | None = Security(_admin_header)) -> None:
    """Guards flight management and RAG rebuild. Disabled entirely until ADMIN_API_KEY is set."""
    if not settings.admin_api_key:
        raise HTTPException(403, "Admin endpoints are disabled. Set ADMIN_API_KEY in .env.")
    if not key or not secrets.compare_digest(key, settings.admin_api_key):
        raise HTTPException(401, "Invalid or missing X-Admin-Key header")