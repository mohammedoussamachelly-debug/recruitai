"""
rag.py — Semantic candidate grouping using Qdrant vector search.
Falls back to in-memory cosine similarity if Qdrant is unavailable.
"""
import os
import numpy as np
from core.vectorize import (
    COLLECTION_NAME,
    get_embedding,
    get_embedding_model,
    get_qdrant_client,
    vectorize_candidates,
)

# Domain semantic queries (French + English vocabulary for better matching)
DOMAIN_QUERIES = {
    "Formation & Éducation": (
        "éducation formation enseignement pédagogie tuteur formateur coach atelier "
        "apprentissage curriculum orientation mentorat facilitateur"
    ),
    "Social & Solidarité": (
        "travail social bénévolat solidarité humanitaire inclusion communauté engagement "
        "citoyen droits migrants réfugiés association santé bien-être"
    ),
    "Environnement & Développement": (
        "environnement écologie développement durable géomatique cartographie territoire "
        "conservation biodiversité énergie renouvelable agriculture"
    ),
    "Culture & Arts": (
        "culture arts musique théâtre cinéma photographie design patrimoine festival "
        "animation culturelle illustration créatif audiovisuel"
    ),
    "Numérique & Innovation": (
        "numérique informatique développeur data intelligence artificielle innovation "
        "startup technologie programmation cybersécurité machine learning"
    ),
    "Communication & Médias": (
        "communication journalisme réseaux sociaux community management relations publiques "
        "presse marketing storytelling branding contenu"
    ),
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _normalize_text(text: str) -> str:
    return str(text or "").lower()


def _group_keyword_fallback(candidates: list, domains: list, per_group: int) -> dict:
    """Fast lexical fallback that avoids loading embedding models."""
    groups = {d: [] for d in domains}
    assigned: set[int] = set()

    normalized_candidate_texts: list[str] = []
    for c in candidates:
        info = c.get("info", {})
        txt = " ".join([
            info.get("name", ""),
            info.get("summary", "") or "",
            info.get("department", "") or "",
            " ".join(info.get("skills", []) if isinstance(info.get("skills"), list) else []),
            c.get("cv_text", "")[:4000],
        ])
        normalized_candidate_texts.append(_normalize_text(txt))

    for domain in domains:
        base = DOMAIN_QUERIES.get(domain, domain)
        keywords = [kw.strip() for kw in _normalize_text(base).split() if len(kw.strip()) > 2]
        if not keywords:
            keywords = [w for w in _normalize_text(domain).split() if len(w) > 2]

        scored = []
        for i, txt in enumerate(normalized_candidate_texts):
            if i in assigned:
                continue
            score = 0.0
            for kw in keywords:
                score += txt.count(kw)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, i in scored[:per_group]:
            groups[domain].append({"candidate": candidates[i], "score": float(score), "matched_kw": []})
            assigned.add(i)

    leftover = [
        {"candidate": candidates[i], "score": 0.0, "matched_kw": []}
        for i in range(len(candidates)) if i not in assigned
    ]
    if leftover:
        groups["Non classifié"] = leftover
    return {k: v for k, v in groups.items() if v}


def _group_inmemory(candidates: list, domains: list, per_group: int) -> dict:
    """Cosine similarity grouping — used when Qdrant is unavailable."""
    try:
        model = get_embedding_model()
    except Exception:
        return _group_keyword_fallback(candidates, domains, per_group)

    cv_vecs = []
    try:
        for c in candidates:
            info = c.get("info", {})
            text = " ".join([
                info.get("name", ""),
                info.get("summary", "") or "",
                info.get("department", "") or "",
                " ".join(info.get("skills", []) if isinstance(info.get("skills"), list) else []),
                c.get("cv_text", "")[:4000],
            ])
            cv_vecs.append(model.encode(text))
    except Exception:
        return _group_keyword_fallback(candidates, domains, per_group)

    groups = {d: [] for d in domains}
    assigned: set = set()

    for domain in domains:
        try:
            query_vec = model.encode(DOMAIN_QUERIES.get(domain, domain))
        except Exception:
            return _group_keyword_fallback(candidates, domains, per_group)
        scored = [
            (_cosine_similarity(query_vec, cv_vecs[i]), i, candidates[i])
            for i in range(len(candidates)) if i not in assigned
        ]
        scored.sort(reverse=True)
        for score, i, c in scored[:per_group]:
            groups[domain].append({"candidate": c, "score": score, "matched_kw": []})
            assigned.add(i)

    leftover = [
        {"candidate": candidates[i], "score": 0, "matched_kw": []}
        for i in range(len(candidates)) if i not in assigned
    ]
    if leftover:
        groups["Non classifié"] = leftover
    return {k: v for k, v in groups.items() if v}


def group_candidates_rag(candidates: list, domains: list, per_group: int) -> dict:
    """
    Group candidates by domain using RAG:
      1. Embed CVs → store in Qdrant
      2. Embed each domain query → retrieve top-k candidates from Qdrant
      3. Assign greedily (no duplicates), leftovers → 'Non classifié'
    Falls back to in-memory cosine similarity if Qdrant is unavailable.
    """
    if not candidates:
        return {}

    # Default to fast/offline-safe behavior. Set ENABLE_SEMANTIC_RAG=1 to force semantic mode.
    if os.getenv("ENABLE_SEMANTIC_RAG", "0") != "1":
        return _group_keyword_fallback(candidates, domains, per_group)

    client = get_qdrant_client()
    if client is None:
        return _group_inmemory(candidates, domains, per_group)

    try:
        vectorize_candidates(candidates)
    except Exception:
        return _group_inmemory(candidates, domains, per_group)

    idx_to_candidate = {i: c for i, c in enumerate(candidates)}
    groups = {d: [] for d in domains}
    assigned: set = set()

    for domain in domains:
        query_vec = get_embedding(DOMAIN_QUERIES.get(domain, domain))
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vec,
                limit=min(per_group * 4, len(candidates)),
            ).points
        except Exception:
            continue

        count = 0
        for hit in results:
            if count >= per_group:
                break
            idx = hit.payload.get("idx", hit.id)
            if idx in assigned:
                continue
            c = idx_to_candidate.get(idx)
            if c is None:
                continue
            groups[domain].append({"candidate": c, "score": hit.score, "matched_kw": []})
            assigned.add(idx)
            count += 1

    leftover = [
        {"candidate": c, "score": 0, "matched_kw": []}
        for i, c in idx_to_candidate.items() if i not in assigned
    ]
    if leftover:
        groups["Non classifié"] = leftover
    return {k: v for k, v in groups.items() if v}
