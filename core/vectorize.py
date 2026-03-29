"""
vectorize.py — Embed CVs and store vectors in Qdrant.
"""
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

COLLECTION_NAME = "cv_profiles"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

_embedding_model: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None
_qdrant_checked = False


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_qdrant_client() -> QdrantClient | None:
    global _qdrant_client, _qdrant_checked
    if _qdrant_checked:
        return _qdrant_client
    _qdrant_checked = True
    host = os.getenv("QDRANT_HOST", "")
    api_key = os.getenv("QDRANT_API_KEY", "")
    if not host or not api_key:
        return None
    try:
        _qdrant_client = QdrantClient(url=host, api_key=api_key)
    except Exception:
        _qdrant_client = None
    return _qdrant_client


def get_embedding(text: str) -> list:
    return get_embedding_model().encode(text).tolist()


def vectorize_candidates(candidates: list):
    """
    Embed each CV and upsert vectors into Qdrant collection 'cv_profiles'.
    Recreates the collection on every call to keep data fresh.
    """
    client = get_qdrant_client()
    if client is None or not candidates:
        return

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = []
    for i, c in enumerate(candidates):
        info = c.get("info", {})
        text = " ".join([
            info.get("name", ""),
            info.get("summary", "") or "",
            info.get("department", "") or "",
            " ".join(info.get("skills", []) if isinstance(info.get("skills"), list) else []),
            c.get("cv_text", "")[:4000],
        ])
        points.append(PointStruct(
            id=i,
            vector=get_embedding(text),
            payload={"filename": c.get("filename", ""), "idx": i},
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
