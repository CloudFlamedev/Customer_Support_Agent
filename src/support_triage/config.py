from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    triage_model: str = "groq/openai/gpt-oss-20b"
    writer_model: str = "groq/openai/gpt-oss-120b"
    guardrail_input_model: str = "groq/meta-llama/llama-prompt-guard-2-22m"
    guardrail_output_model: str = "groq/openai/gpt-oss-safeguard-20b"

    embedding_backend: str = "hf_api"   # "hf_api" | "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token: str = ""

    kb_path: str = "data/kb"
    chroma_path: str = ".chroma"

    confidence_threshold: float = 0.75          # final combined score
    retrieval_floor: float = 0.30               # best KB hit below this = no answer
    retrieval_good: float = 0.60                # best KB hit at/above this = strong
    always_escalate_categories: list[str] = ["legal", "security", "refund_dispute"]


settings = Settings()