"""
cv_analyzer.py — LLM-based CV information extraction and department classification.
Uses GPT-4o-mini via the GitHub Models API.
"""

import os
import json
import re
import unicodedata
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
# Use OpenRouter if available, otherwise fall back to GitHub Models
if OPENROUTER_KEY:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = OPENROUTER_KEY
    API_MODEL = "openai/gpt-4o-mini"
else:
    API_URL = "https://models.inference.ai.azure.com/chat/completions"
    API_KEY = GITHUB_TOKEN
    API_MODEL = "gpt-4o-mini"

# ─── Department keyword map ────────────────────────────────────────────────────
DEPARTMENT_KEYWORDS = {
    "IT": [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "node", "sql", "nosql", "mongodb", "postgresql", "docker", "kubernetes",
        "aws", "azure", "gcp", "cloud", "devops", "ci/cd", "git", "linux",
        "machine learning", "deep learning", "ai", "artificial intelligence",
        "data science", "software", "developer", "programmer", "backend",
        "frontend", "full stack", "fullstack", "api", "microservices",
        "informatique", "developpement", "developpeur", "ingenieur logiciel",
        "science des donnees", "donnees", "ia", "intelligence artificielle",
        "apprentissage automatique", "reseaux", "systemes", "base de donnees",
    ],
    "Marketing": [
        "marketing", "seo", "sem", "social media", "content", "branding",
        "advertising", "campaign", "analytics", "google ads", "facebook ads",
        "copywriting", "email marketing", "crm", "hubspot",
        "communication", "marketing digital", "strategie marketing", "marque",
        "publicite", "campagne", "community management",
    ],
    "Finance": [
        "finance", "accounting", "audit", "tax", "financial analysis",
        "budget", "forecasting", "excel", "sap", "erp", "investment",
        "comptabilite", "controle de gestion", "tresorerie", "fiscalite",
        "analyse financiere", "banque", "assurance",
    ],
    "HR": [
        "human resources", "hr", "recruitment", "talent acquisition",
        "onboarding", "training", "compensation", "benefits", "payroll",
        "ressources humaines", "rh", "paie", "sirh", "sap hcm",
        "gestion des talents", "relations sociales", "droit du travail",
        "formation", "recrutement", "developpement des competences",
    ],
    "Engineering": [
        "mechanical", "electrical", "civil", "structural", "chemical",
        "manufacturing", "cad", "autocad", "solidworks", "catia",
        "engineering", "project management", "pmp",
        "genie mecanique", "genie civil", "genie electrique", "ingenierie",
        "maintenance", "qualite", "production industrielle",
    ],
}


_api_rate_limited = False  # circuit breaker: skip LLM after first 429


def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(
        total=1,
        backoff_factor=0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        respect_retry_after_header=False,  # never sleep on Retry-After
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def classify_department(skills_text: str) -> str:
    if not isinstance(skills_text, str):
        return "General"

    text_lower = _normalize_ascii(skills_text)
    scores = {}

    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        score = 0
        for kw in keywords:
            kw_norm = _normalize_ascii(kw)
            if not kw_norm:
                continue
            score += text_lower.count(kw_norm)
        scores[dept] = score

    if not scores:
        return "General"
        
    best_dept = max(scores, key=scores.get)
    return best_dept if scores[best_dept] > 0 else "General"


def _extract_role_line(cv_text: str) -> str:
    lines = [line.strip() for line in cv_text.splitlines() if line.strip()]
    if not lines:
        return ""

    # Typical CV order: contact line first, role/title in the next lines.
    for line in lines[:12]:
        norm = _normalize_ascii(line)
        if "@" in line:
            continue
        if _looks_like_role_or_section(line):
            # Keep role lines and skip obvious section headings.
            if any(
                token in norm
                for token in [
                    "developer", "ingenieur", "manager", "responsable", "scientist",
                    "analyst", "consultant", "rh", "ressources humaines", "marketing",
                    "finance", "accountant", "devops", "data",
                ]
            ):
                return line
            continue
        if 2 <= len(line.split()) <= 8 and len(line) <= 70:
            return line
    return ""


def _classify_department_from_cv(cv_text: str, skills: list, summary: str, llm_department: str = "") -> str:
    text_full = _normalize_ascii(cv_text)
    title_line = _normalize_ascii(_extract_role_line(cv_text))
    skills_text = _normalize_ascii(" ".join(skills) if isinstance(skills, list) else "")
    summary_text = _normalize_ascii(summary if isinstance(summary, str) else "")

    scores = {dept: 0.0 for dept in DEPARTMENT_KEYWORDS.keys()}

    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        for kw in keywords:
            kw_norm = _normalize_ascii(kw)
            if not kw_norm:
                continue
            # Weighted signals: title > skills > summary > full text.
            scores[dept] += title_line.count(kw_norm) * 6.0
            scores[dept] += skills_text.count(kw_norm) * 4.0
            scores[dept] += summary_text.count(kw_norm) * 2.5
            scores[dept] += text_full.count(kw_norm) * 1.0

    # Mild prior from model output if it matches known departments.
    if isinstance(llm_department, str) and llm_department in scores:
        scores[llm_department] += 1.5

    best_dept = max(scores, key=scores.get)
    if scores[best_dept] <= 0:
        return "General"
    return best_dept


def _safe_float(value, default=0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return default


def _normalize_ascii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    ).lower()


def _looks_like_role_or_section(text: str) -> bool:
    if not isinstance(text, str):
        return True
    lower = _normalize_ascii(text).strip()
    role_tokens = [
        "developer",
        "engineer",
        "scientist",
        "manager",
        "responsable",
        "consultant",
        "intern",
        "stagiaire",
        "full stack",
        "backend",
        "front end",
        "frontend",
        "analyst",
        "lead",
        "architect",
        "specialist",
        "officer",
        "director",
        "head of",
        "product",
        "project",
        "hr",
        "resources humaines",
    ]
    section_tokens = [
        "profile",
        "summary",
        "experience",
        "experience professionnelle",
        "parcours professionnel",
        "education",
        "formation",
        "skills",
        "competences",
        "competences techniques",
        "projets",
        "certifications",
        "curriculum vitae",
        "cv",
    ]
    return any(token in lower for token in role_tokens + section_tokens)


def _looks_like_person_name(text: str) -> bool:
    if not isinstance(text, str):
        return False
    raw = text.strip()
    if not raw:
        return False

    normalized = _normalize_ascii(raw)
    if normalized in {"unknown candidate", "unknown", "n/a", "na", "none"}:
        return False
    if _looks_like_role_or_section(raw):
        return False
    if any(ch.isdigit() for ch in raw):
        return False
    if len(raw) < 4 or len(raw) > 60:
        return False

    # Reject common non-name fragments frequently found in CV headings/content.
    forbidden_tokens = {
        "institut",
        "universite",
        "university",
        "ecole",
        "master",
        "licence",
        "bachelor",
        "phd",
        "doctorat",
        "etudiant",
        "etudiante",
        "ingenieur",
        "ingenieure",
        "recherche",
        "environnement",
        "foresterie",
        "genie",
        "data",
        "science",
        "territoire",
        "amenagement",
        "direction",
        "generale",
        "forets",
        "juin",
    }
    normalized_words = set(normalized.split())
    if normalized_words & forbidden_tokens:
        return False

    parts = [p for p in re.split(r"\s+", raw) if p]
    if len(parts) < 2 or len(parts) > 5:
        return False
    if not re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ'\- ]+$", raw):
        return False

    # Reject obvious heading-style lines (all uppercase long tokens).
    if raw.isupper() and len(parts) >= 2:
        heading_words = {
            "experience",
            "professionnelle",
            "education",
            "formation",
            "competences",
            "skills",
            "summary",
            "profile",
            "objectif",
            "curriculum",
            "vitae",
        }
        if any(word in normalized.split() for word in heading_words):
            return False

    return True


def _extract_name_from_email_line(line: str) -> str:
    if not isinstance(line, str) or "@" not in line:
        return ""
    match = re.match(
        r"^\s*([A-Za-zÀ-ÖØ-öø-ÿ'\- ]{3,60})\s+[\w.+-]+@[\w-]+\.[\w.-]+",
        line.strip(),
    )
    if not match:
        return ""
    candidate = " ".join(match.group(1).split())
    return candidate if _looks_like_person_name(candidate) else ""


def _name_from_email(email: str) -> str:
    if not isinstance(email, str) or "@" not in email:
        return ""
    local_part = email.split("@", 1)[0].strip().lower()
    local_part = re.sub(r"\d+", "", local_part)
    chunks = [chunk for chunk in re.split(r"[._\-]+", local_part) if chunk]
    if len(chunks) < 2:
        return ""
    words = []
    for chunk in chunks[:4]:
        if len(chunk) < 2:
            continue
        words.append(chunk.capitalize())
    return " ".join(words)


def _extract_name_from_text(cv_text: str) -> str:
    lines = [line.strip() for line in cv_text.splitlines() if line.strip()]

    # Common CV format: "Full Name email@domain | phone | location"
    for line in lines[:25]:
        from_email_line = _extract_name_from_email_line(line)
        if from_email_line:
            return from_email_line

    for line in lines[:20]:
        lower = line.lower()
        # Skip lines that are very likely section titles or contact labels.
        if any(
            token in lower
            for token in [
                "email",
                "phone",
                "address",
                "linkedin",
                "github",
                "profile",
                "summary",
                "experience",
                "education",
                "skills",
                "curriculum",
                "vitae",
            ]
        ):
            continue
        if _looks_like_person_name(line):
            return " ".join(line.split())
    return "Unknown Candidate"


def _name_from_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        return ""

    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]

    # Remove common CV prefixes and separators.
    base = re.sub(r"(?i)\bcv\b", " ", base)
    base = re.sub(r"[._\-]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    base = re.sub(r"\b\d{4}\b", " ", base)
    base = re.sub(r"\s+", " ", base).strip(" .")

    if not base:
        return ""

    parts = []
    for token in base.split():
        token = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'\-]", "", token)
        if not token:
            continue
        low = _normalize_ascii(token)
        if low in {"pdf", "doc", "docx", "final", "version", "v"}:
            continue
        if any(ch.isdigit() for ch in token):
            continue
        if len(token) <= 1:
            continue
        parts.append(token.capitalize())

    # Remove duplicated halves like "Hana Trigui Hana Trigui".
    if len(parts) >= 4 and len(parts) % 2 == 0:
        half = len(parts) // 2
        if parts[:half] == parts[half:]:
            parts = parts[:half]

    # Handle mirrored patterns like "Aya Nalouti Nalouti Aya".
    if len(parts) == 4 and parts[0] == parts[3] and parts[1] == parts[2]:
        parts = parts[:2]

    candidate = " ".join(parts[:5]).strip()
    return candidate if _looks_like_person_name(candidate) else ""


def _extract_education_from_text(cv_text: str) -> str:
    patterns = [
        r"(?im)^(?:education|formation|academic background)\s*[:\-]?\s*(.+)$",
        r"(?im)\b(?:bachelor|master|phd|doctorate|engineer|licence|diploma|mba)\b[^\n]{0,120}",
    ]
    for pattern in patterns:
        match = re.search(pattern, cv_text)
        if match:
            text = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if text:
                return text
    return "Not specified"


def _extract_experience_from_text(cv_text: str) -> float:
    normalized_text = _normalize_ascii(cv_text)
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|ans|an)\s+(?:of\s+)?experience",
        r"(?:avec\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:ans|an)\s+d[' ]?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
        if match:
            return _safe_float(match.group(1), 0.0)

    # Fallback: estimate experience from year spans in the CV timeline.
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", normalized_text)]
    current_year = 2026
    years = [y for y in years if 1970 <= y <= current_year]
    if len(years) >= 2:
        span = max(years) - min(years)
        if 0 < span <= 45:
            return float(span)
    return 0.0


def _extract_phone_from_text(cv_text: str) -> str:
    if not isinstance(cv_text, str):
        return ""

    candidates = re.findall(r"(?:\+\d{1,3}[\s().-]*)?(?:\d[\s().-]*){8,14}", cv_text)
    for raw in candidates:
        cleaned = " ".join(raw.split()).strip(" -./")
        # Remove trailing year chunks accidentally merged with phone numbers.
        cleaned = re.sub(r"\s+(?:19|20)\d{2}$", "", cleaned)
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) < 8 or len(digits) > 15:
            continue
        # Reject date-like values such as 2024-2025.
        if re.search(r"\b(?:19|20)\d{2}\s*[-/]\s*(?:19|20)\d{2}\b", cleaned):
            continue
        if re.search(r"\b(?:19|20)\d{2}\b", cleaned) and len(digits) <= 8:
            continue
        return cleaned
    return ""


def _normalize_parsed_info(parsed: dict, cv_text: str, source_name: str = "") -> dict:
    normalized = parsed if isinstance(parsed, dict) else {}

    name = (
        normalized.get("name")
        or normalized.get("nom")
        or normalized.get("full_name")
        or _extract_name_from_text(cv_text)
    )
    email = normalized.get("email", "")
    phone = normalized.get("phone", "") or _extract_phone_from_text(cv_text)

    skills = normalized.get("skills", [])
    if isinstance(skills, str):
        skills = [skills]
    elif not isinstance(skills, list):
        skills = []

    experience = _safe_float(
        normalized.get("years_experience", normalized.get("experience", 0)),
        _extract_experience_from_text(cv_text),
    )
    education = (
        normalized.get("education")
        or normalized.get("degree")
        or normalized.get("highest_degree")
        or _extract_education_from_text(cv_text)
    )

    languages = normalized.get("languages", [])
    if isinstance(languages, str):
        languages = [languages]
    elif not isinstance(languages, list):
        languages = []

    summary = normalized.get("summary", "")

    email_name = _name_from_email(email)
    file_name = _name_from_filename(source_name)

    def _prefer_file_name(current_name: str, candidate_file_name: str) -> bool:
        if not candidate_file_name:
            return False
        if not isinstance(current_name, str) or not current_name.strip():
            return True
        name_clean = current_name.strip()
        # OCR names are often all-caps, clipped, or contain replacement chars.
        if "�" in name_clean:
            return True
        if name_clean.isupper() and len(name_clean.split()) >= 2:
            return True
        if len(name_clean.split()) == 2 and any(len(part) <= 2 for part in name_clean.split()):
            return True
        return False

    # If name looks like a job title/section, prefer a better fallback.
    if _prefer_file_name(name, file_name):
        name = file_name
    elif not _looks_like_person_name(name):
        text_name = _extract_name_from_text(cv_text)
        # Prefer filename when text-derived names look suspicious in noisy PDFs.
        if file_name:
            name = file_name
        elif _looks_like_person_name(text_name):
            name = text_name
        elif email_name:
            name = email_name
        else:
            name = "Unknown Candidate"

    enriched = {
        "name": name if isinstance(name, str) and name.strip() else "Unknown Candidate",
        "email": email if isinstance(email, str) else "",
        "phone": phone if isinstance(phone, str) else "",
        "skills": [str(s).strip() for s in skills if str(s).strip()],
        "years_experience": round(experience, 1),
        "education": str(education).strip() if education else "Not specified",
        "languages": [str(l).strip() for l in languages if str(l).strip()],
        "summary": str(summary).strip() if summary else "",
    }

    enriched["department"] = _classify_department_from_cv(
        cv_text=cv_text,
        skills=enriched["skills"],
        summary=enriched["summary"],
        llm_department=str(normalized.get("department", "")),
    )
    return enriched


def extract_cv_info(cv_text: str, source_name: str = "") -> dict:
    global _api_rate_limited
    if not API_KEY or _api_rate_limited:
        return _fallback_info(cv_text, "No API key or rate limited", source_name=source_name)

    prompt = """You are a recruitment AI. Analyze the following CV text.
Return ONLY a valid JSON object. No explanation, no intro, no markdown text blocks around the JSON.

{
    "name": "Full name of the candidate (check both 'name' and 'nom')",
    "email": "Email address or empty string",
    "phone": "Phone number or empty string",
    "skills": ["list", "of", "skills"],
    "years_experience": 0,
    "education": "Highest degree and institution",
    "languages": ["list", "of", "languages"],
    "summary": "Brief 2-sentence summary"
}

CV TEXT:
"""

    try:
        session = create_session_with_retries()
        response = session.post(
            url=API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": API_MODEL,
                "messages": [
                    {"role": "system", "content": "You extract structured data from CVs. Respond with valid JSON only."},
                    {"role": "user", "content": prompt + cv_text[:4000]},
                ],
                "max_tokens": 800,
                "temperature": 0.1,
            },
            timeout=(8, 30),
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)
            return _normalize_parsed_info(parsed, cv_text, source_name=source_name)
        else:
            if response.status_code == 429:
                _api_rate_limited = True  # trip circuit breaker for all subsequent calls
            return _fallback_info(cv_text, f"API error {response.status_code}", source_name=source_name)

    except json.JSONDecodeError as e:
        return _fallback_info(cv_text, f"JSON Error: {str(e)}", source_name=source_name)
    except Exception as e:
        return _fallback_info(cv_text, f"Error: {str(e)}", source_name=source_name)


def _fallback_info(cv_text: str, error_msg: str, source_name: str = "") -> dict:
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", cv_text)

    fallback = {
        "name": _extract_name_from_text(cv_text) or _name_from_filename(source_name) or "Unknown Candidate",
        "email": email_match.group(0) if email_match else "",
        "phone": _extract_phone_from_text(cv_text),
        "skills": [],
        "years_experience": _extract_experience_from_text(cv_text),
        "education": _extract_education_from_text(cv_text),
        "languages": [],
        "summary": f"(Auto-extraction failed: {error_msg})",
    }
    return _normalize_parsed_info(fallback, cv_text, source_name=source_name)
