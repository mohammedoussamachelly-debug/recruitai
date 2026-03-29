"""
server.py - RecruitAI web backend (API + static frontend)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import os
import threading

from core.analyzer import extract_cv_info
from core.parser import parse_cv
from core.rag import group_candidates_rag

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
_cv_cache: dict = {}  # key: (filename, mtime) → candidate dict


def _preload_model() -> None:
    """Warm up the embedding model in the background at startup."""
    if os.getenv("ENABLE_SEMANTIC_RAG", "0") == "1":
        try:
            from core.vectorize import get_embedding_model
            get_embedding_model()
        except Exception:
            pass


threading.Thread(target=_preload_model, daemon=True).start()
WEB_DIR = BASE_DIR / "web"
CV_DIR = BASE_DIR / "cvs"

PRESET_DOMAINS = [
    {"id": "formation", "label": "Formation et Education", "icon": "🎓", "color": "#6366F1"},
    {"id": "social", "label": "Social et Solidarite", "icon": "🤝", "color": "#EC4899"},
    {"id": "environnement", "label": "Environnement et Developpement", "icon": "🌿", "color": "#10B981"},
    {"id": "culture", "label": "Culture et Arts", "icon": "🎨", "color": "#F59E0B"},
    {"id": "numerique", "label": "Numerique et Innovation", "icon": "💻", "color": "#3B82F6"},
    {"id": "communication", "label": "Communication et Medias", "icon": "📢", "color": "#8B5CF6"},
]

EXTRA_COLORS = ["#F97316", "#14B8A6", "#84CC16", "#EF4444", "#06B6D4"]

# Domain vocabulary used to generate per-candidate keyword explanations
DOMAIN_VOCAB: dict[str, list[str]] = {
    "Formation et Education": [
        "formation", "education", "enseignement", "pedagogie", "tuteur", "formateur",
        "coach", "apprentissage", "mentorat", "atelier", "curriculum", "orientation",
        "teaching", "training", "learning", "instructor", "mentor",
    ],
    "Social et Solidarite": [
        "social", "solidarite", "benevole", "humanitaire", "inclusion", "communaute",
        "citoyen", "droits", "migrants", "refugies", "association", "sante", "bien-etre",
        "volunteer", "community", "humanitarian", "solidarity",
    ],
    "Environnement et Developpement": [
        "environnement", "ecologie", "developpement", "durable", "conservation",
        "biodiversite", "energie", "agriculture", "territoire", "cartographie",
        "geomatique", "environment", "ecology", "sustainable", "renewable",
    ],
    "Culture et Arts": [
        "culture", "arts", "musique", "theatre", "cinema", "photographie", "design",
        "patrimoine", "festival", "animation", "illustration", "audiovisuel",
        "music", "art", "creative", "photography",
    ],
    "Numerique et Innovation": [
        "numerique", "informatique", "developpeur", "data", "intelligence", "artificielle",
        "innovation", "technologie", "programmation", "cybersecurite", "machine", "learning",
        "startup", "digital", "developer", "software", "python", "javascript", "react",
        "web", "api", "cloud", "devops", "ia", "ai",
    ],
    "Communication et Medias": [
        "communication", "journalisme", "reseaux", "sociaux", "community", "management",
        "relations", "publiques", "presse", "marketing", "storytelling", "branding",
        "contenu", "media", "publicite", "redaction", "editorial",
    ],
}


class AnalyzeRequest(BaseModel):
    domains: list[str] = Field(default_factory=list)
    custom_domain: str | None = None
    per_group: int | None = Field(default=3, ge=1, le=2000)
    per_group_text: str | None = None


app = FastAPI(title="RecruitAI Web App", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def _find_matched_kw(domain: str, cv_text: str) -> list[str]:
    """Find domain vocabulary words that actually appear in the CV text."""
    vocab = DOMAIN_VOCAB.get(domain, [])
    cv_norm = _norm(cv_text)
    seen: set[str] = set()
    matched: list[str] = []
    for kw in vocab:
        kw_norm = _norm(kw)
        if kw_norm and kw_norm not in seen and kw_norm in cv_norm:
            seen.add(kw_norm)
            matched.append(kw)
        if len(matched) >= 5:
            break
    return matched


def _norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch)
    )


def _get_domain_meta(label: str, custom: list[str]) -> tuple[str, str]:
    for d in PRESET_DOMAINS:
        if d["label"] == label:
            return d["color"], d["icon"]
    idx = custom.index(label) % len(EXTRA_COLORS) if label in custom else 0
    return EXTRA_COLORS[idx], "GEN"


def _group_summary(domain: str, entries: list[dict]) -> str:
    if _norm(domain) in {"non classe", "non classifie", "non classifiee"}:
        return (
            "Ces profils n'ont pas pu etre alignes avec les domaines selectionnes. "
            "Ils peuvent servir de renfort polyvalent."
        )

    all_kw: list[str] = []
    for e in entries:
        all_kw.extend(e.get("matched_kw", []))

    top = [kw for kw, _ in Counter(all_kw).most_common(5)]
    if top:
        kw_str = ", ".join(top)
        return (
            f"Ce groupe est forme selon les mots-cles les plus proches du domaine: {kw_str}."
        )

    return "Ce groupe a ete construit par similarite semantique des CV avec le domaine choisi."


def _candidate_payload(entry: dict) -> dict:
    info = entry.get("candidate", {}).get("info", {})
    raw_skills = info.get("skills", [])
    skills = raw_skills[:6] if isinstance(raw_skills, list) else []

    raw_summary = info.get("summary", "") or ""
    clean_summary = "" if raw_summary.startswith("(Auto-extraction") else raw_summary

    return {
        "name": info.get("name", "Inconnu"),
        "email": info.get("email", ""),
        "phone": info.get("phone", ""),
        "education": info.get("education", ""),
        "experience": info.get("years_experience", 0),
        "department": info.get("department", "General"),
        "skills": skills,
        "score": round(float(entry.get("score", 0) or 0), 4),
        "matched_kw": entry.get("matched_kw", []) or [],
        "summary": clean_summary,
    }


def _parse_per_group(req: AnalyzeRequest) -> tuple[int, str]:
    if req.per_group_text:
        raw = req.per_group_text.strip()
        if raw:
            m = re.search(r"\d+", raw)
            return (int(m.group()) if m else 999), raw
    val = int(req.per_group or 3)
    return val, f"{val} personnes par groupe"


def _load_one_cv(f: Path) -> tuple[dict | None, str | None]:
    """Parse and analyze a single CV. Results are cached by (filename, mtime)."""
    try:
        mtime = f.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = (f.name, mtime)
    if key in _cv_cache:
        return _cv_cache[key], None
    try:
        cv_text = parse_cv(str(f), f.name)
        if not cv_text.strip():
            return None, f"{f.name}: fichier vide ou illisible"
        info = extract_cv_info(cv_text, source_name=f.name)
        result = {"filename": f.name, "cv_text": cv_text, "info": info}
        _cv_cache[key] = result
        return result, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{f.name}: {exc}"


def _load_all_cvs() -> tuple[list[dict], list[str]]:
    candidates: list[dict] = []
    errors: list[str] = []

    if not CV_DIR.exists():
        return candidates, [f"Dossier cvs introuvable: {CV_DIR}"]

    files = list(chain(CV_DIR.glob("*.pdf"), CV_DIR.glob("*.docx")))
    if not files:
        return candidates, []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_load_one_cv, f): f for f in files}
        for future in as_completed(futures):
            candidate, error = future.result()
            if candidate:
                candidates.append(candidate)
            if error:
                errors.append(error)

    return candidates, errors


@app.get("/")
def index() -> FileResponse:
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend file web/index.html not found")
    return FileResponse(index_file)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/domains")
def get_domains() -> dict:
    return {"domains": PRESET_DOMAINS}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    selected_domains = [d.strip() for d in request.domains if isinstance(d, str) and d.strip()]

    custom_domains: list[str] = []
    if request.custom_domain and request.custom_domain.strip():
        custom_label = request.custom_domain.strip()
        if custom_label not in selected_domains:
            selected_domains.append(custom_label)
        custom_domains.append(custom_label)

    if not selected_domains:
        raise HTTPException(status_code=400, detail="Selectionnez au moins un domaine")

    per_group, per_group_label = _parse_per_group(request)

    candidates, errors = _load_all_cvs()
    groups = group_candidates_rag(candidates, selected_domains, per_group)

    # Enrich matched_kw for each entry so explanations work regardless of RAG mode
    for domain, entries in groups.items():
        for entry in entries:
            if not entry.get("matched_kw"):
                cv_text = entry.get("candidate", {}).get("cv_text", "")
                entry["matched_kw"] = _find_matched_kw(domain, cv_text)

    serialized_groups: list[dict] = []
    for domain, entries in groups.items():
        color, icon = _get_domain_meta(domain, custom_domains)
        serialized_groups.append(
            {
                "domain": domain,
                "color": color,
                "icon": icon,
                "count": len(entries),
                "summary": _group_summary(domain, entries),
                "candidates": [_candidate_payload(e) for e in entries],
            }
        )

    total = sum(group["count"] for group in serialized_groups)

    return {
        "stats": {
            "total_candidates": total,
            "group_count": len(serialized_groups),
            "max_per_group": "max" if per_group >= 999 else per_group,
            "per_group_label": per_group_label,
        },
        "groups": serialized_groups,
        "errors": errors,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
