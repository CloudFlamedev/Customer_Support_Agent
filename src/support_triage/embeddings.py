"""One place that decides how text becomes vectors.

Ingestion and search MUST use the same model, otherwise
similarity scores are meaningless.
"""
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from support_triage.config import settings


class HFInferenceEmbedding(EmbeddingFunction):
    """Calls Hugging Face's hosted API (free tier, rate limited)."""

    def __init__(self, model: str, token: str):
        from huggingface_hub import InferenceClient

        if not token:
            raise ValueError("HF_TOKEN is missing. Add it to your .env file.")
        self.model = model
        self.client = InferenceClient(provider="hf-inference", api_key=token)

    @staticmethod
    def name() -> str:
        return "hf_inference_embedding"

    # Free tier can return 503 (model loading) or 429 (rate limit): wait and retry
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=20))
    def _embed_one(self, text: str) -> list[float]:
        out = np.asarray(self.client.feature_extraction(text, model=self.model))
        if out.ndim > 1:            # some models return one row per token
            out = out.mean(axis=0)  # average them into one sentence vector
        norm = np.linalg.norm(out)
        return (out / norm if norm else out).astype(float).tolist()

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(t) for t in input]


def get_embedding_function():
    if settings.embedding_backend == "hf_api":
        return HFInferenceEmbedding(settings.embedding_model, settings.hf_token)

    from chromadb.utils import embedding_functions  # local fallback (needs sentence-transformers)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )