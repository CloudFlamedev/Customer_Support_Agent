"""Knowledge base search.

search_kb()      -> plain Python function. Our Flow calls this directly
                    to get the retrieval score (decisions stay in code).
KBSearchTool     -> thin wrapper so a CrewAI agent can call the same search.
"""
from dataclasses import dataclass

import chromadb
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from support_triage.config import settings
from support_triage.embeddings import get_embedding_function

COLLECTION_NAME = "support_kb"


@dataclass
class KBHit:
    text: str
    source: str
    score: float  # 0 to 1, higher = closer match


def get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_path)
    # We create embeddings ourselves, so no embedding_function is attached here.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def search_kb(query: str, k: int = 3) -> list[KBHit]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vec = get_embedding_function()([query])
    result = collection.query(
        query_embeddings=query_vec,
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        # Chroma cosine distance = 1 - similarity
        hits.append(KBHit(text=text, source=meta["source"], score=max(0.0, 1.0 - dist)))
    return hits


class KBSearchInput(BaseModel):
    query: str = Field(description="The customer's question, in plain words")


class KBSearchTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "Search the company knowledge base (policies, FAQs). "
        "Returns matching passages with their source file names."
    )
    args_schema: type[BaseModel] = KBSearchInput

    def _run(self, query: str) -> str:
        hits = search_kb(query)
        if not hits:
            return "NO_RESULTS"
        return "\n\n".join(
            f"[source: {h.source} | score: {h.score:.2f}]\n{h.text}" for h in hits
        )
