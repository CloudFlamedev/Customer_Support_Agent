"""Load KB files into ChromaDB.  Run:  python -m support_triage.ingest"""
import hashlib
import json
import re
from pathlib import Path

from support_triage.config import settings
from support_triage.embeddings import get_embedding_function
from support_triage.tools.kb_search import get_collection

MAX_CHARS = 700


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split by paragraphs, then merge small ones up to max_chars.
    Every chunk starts with the document title so it keeps its context."""
    title_match = re.match(r"#\s+(.+)", text)
    title = title_match.group(1) if title_match else ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)

    # Add the title to chunks that don't already start with it
    return [c if c.startswith("# ") else f"{title}\n{c}" for c in chunks]


def ingest() -> None:
    kb_dir = Path(settings.kb_path)
    chroma_dir = Path(settings.chroma_path)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = chroma_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}

    collection = get_collection()
    embed = get_embedding_function()
    files = sorted([*kb_dir.glob("*.md"), *kb_dir.glob("*.txt")])
    current_names = {f.name for f in files}

    # Remove chunks of files that were deleted from disk
    for gone in set(manifest) - current_names:
        collection.delete(where={"source": gone})
        manifest.pop(gone)
        print(f"removed  {gone}")

    for f in files:
        text = f.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode()).hexdigest()
        if manifest.get(f.name) == digest:
            print(f"skipped  {f.name} (unchanged)")
            continue

        collection.delete(where={"source": f.name})  # drop old version, if any
        chunks = chunk_text(text)
        collection.add(
            ids=[f"{f.name}::{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embed(chunks),
            metadatas=[{"source": f.name, "chunk": i} for i in range(len(chunks))],
        )
        manifest[f.name] = digest
        print(f"ingested {f.name} ({len(chunks)} chunks)")

    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"done. total chunks in DB: {collection.count()}")


if __name__ == "__main__":
    ingest()
