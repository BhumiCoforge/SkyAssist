"""Central settings (Phase 2: pydantic-settings instead of scattered os.getenv calls)."""
from functools import lru_cache
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from netfix import prefer_ipv4

# langchain-anthropic reads ANTHROPIC_API_KEY from the real environment, so export .env there too.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "SkyAssist API"
    database_url: str = "sqlite:///./skyassist.db"

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-5-5"
    llm_max_tokens: int = 1024
    # Gateway settings (e.g. company LLM router). Leave llm_base_url empty to call the provider directly.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_api_key_header: str = "X-API-KEY"   # header name the gateway expects the key in
    # Some networks resolve the gateway to an IPv6 address that never answers; this skips it (see netfix.py).
    llm_force_ipv4: bool = True

    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 4
    rag_min_score: float = 0.20   # relevance threshold (0-1). Chosen with evals/run_eval.py

    # Business rules
    cancel_fee: int = 500
    seed_days: int = 45

    # Agent
    confirm_ttl_seconds: int = 600       # a pending booking/cancellation expires after 10 minutes
    history_max_tokens: int = 6000       # chat memory sent to the model (older turns are trimmed)
    agent_max_steps: int = 6

    # Protects flight create/update/delete and /rag/rebuild. Empty = those endpoints are disabled.
    admin_api_key: str = ""

    # Razorpay (TEST MODE only for local implementation)
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    payment_order_ttl_minutes: int = 15
    payment_provider: str = "demo"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

if settings.llm_base_url and settings.llm_force_ipv4:
    prefer_ipv4(urlparse(settings.llm_base_url).hostname)