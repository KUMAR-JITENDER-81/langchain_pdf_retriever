from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any

from app.services.ocr_service import normalize_text, ocr_text_quality


TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
GENERAL_QUESTION_PHRASES = (
    "what is this",
    "what's this",
    "whats this",
    "what is the document",
    "what is the pdf",
    "summarize",
    "summary",
    "overview",
    "main idea",
    "main topic",
)
STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "document",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "main",
    "me",
    "of",
    "on",
    "or",
    "pdf",
    "please",
    "say",
    "tell",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
}
ACTION_STARTERS = {
    "achieved",
    "analyzed",
    "applied",
    "architected",
    "built",
    "completed",
    "configured",
    "created",
    "deployed",
    "designed",
    "developed",
    "earned",
    "established",
    "evaluated",
    "followed",
    "implemented",
    "improved",
    "integrated",
    "introduced",
    "led",
    "managed",
    "optimized",
    "qualified",
    "rated",
    "reduced",
    "solved",
    "supported",
    "used",
}
PROFILE_SECTIONS = {
    "Education": ("EDUCATION",),
    "Experience": ("WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EXPERIENCE"),
    "Skills": ("TECHNICAL SKILLS", "COURSEWORK / SKILLS", "SKILLS"),
    "Projects": ("PROJECTS", "PERSONAL PROJECTS"),
    "Achievements": ("ACHIEVEMENTS", "AWARDS"),
    "Certifications": ("CERTIFICATIONS", "CERTIFICATES"),
    "Abstract": ("ABSTRACT",),
    "Introduction": ("INTRODUCTION",),
    "Methodology": ("METHODOLOGY", "METHODS"),
    "Results": ("RESULTS", "FINDINGS"),
    "Conclusion": ("CONCLUSION", "CONCLUSIONS"),
    "References": ("REFERENCES", "BIBLIOGRAPHY"),
    "Executive summary": ("EXECUTIVE SUMMARY",),
    "Recommendations": ("RECOMMENDATIONS",),
    "Installation": ("INSTALLATION", "SETUP"),
    "Troubleshooting": ("TROUBLESHOOTING",),
}


@dataclass(slots=True)
class SentenceCandidate:
    text: str
    source_id: int
    source_relevance: float
    source_position: int
    sentence_position: int
    page_number: int
    tokens: set[str]
    text_quality: float
    category: str
    score: float = 0.0


def generate_local_answer(
    question: str,
    results: list[dict[str, Any]],
    mode: str,
) -> str:
    """Build a grounded answer using only sentences from retrieved PDF excerpts."""
    general_question = is_overview_question(question)
    candidates = _collect_candidates(results, general_question=general_question)
    if not candidates:
        return "I could not find readable evidence for that question in the selected documents."

    query_tokens = set(_content_tokens(question))
    _score_candidates(candidates, query_tokens, general_question)
    sentence_limit = {"quick": 2, "balanced": 5, "deep": 8}.get(mode, 5)
    if general_question and mode == "quick":
        sentence_limit = 3
    selected = _select_diverse(candidates, sentence_limit)
    if not selected:
        return "I could not find sufficiently relevant evidence in the selected documents."

    cited = [f"{candidate.text} [Source {candidate.source_id}]" for candidate in selected]
    if mode == "quick":
        if general_question:
            title = _document_title(results)
            profile = build_document_profile(results)
            if profile["type"]:
                subject = title or "The selected PDF"
                article = "an" if str(profile["type"])[0].lower() in "aeiou" else "a"
                introduction = (
                    f"{subject} appears to be {article} {profile['type']} [Source 1]. "
                    "Its strongest extracted points are:"
                )
            else:
                introduction = (
                    f"{title} appears to cover these main points:"
                    if title
                    else "The selected PDF appears to cover these main points:"
                )
            return introduction + "\n" + "\n".join(
                f"- {sentence}" for sentence in cited
            )
        return " ".join(cited)
    if mode == "deep":
        bullets = "\n".join(f"- {sentence}" for sentence in cited)
        return (
            "## Evidence-based answer\n\n"
            f"{bullets}\n\n"
            "The points above are extracted from the strongest matching passages; "
            "use the source links to verify wording and context."
        )
    return "Based on the selected PDF evidence:\n\n" + "\n".join(
        f"- {sentence}" for sentence in cited
    )


def stream_text(text: str, target_characters: int = 56):
    """Yield small readable chunks so the no-LLM fallback still streams in the UI."""
    buffer = ""
    for piece in re.findall(r"\S+\s*", text):
        buffer += piece
        if len(buffer) >= target_characters or "\n" in piece:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def contextualize_locally(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """Resolve short follow-ups without calling a language model."""
    normalized = " ".join(question.split())
    tokens = _tokens(normalized)
    follow_up_words = {"it", "its", "that", "this", "they", "them", "those", "these"}
    if len(tokens) > 10 and not (set(tokens) & follow_up_words):
        return normalized

    previous_question = next(
        (
            " ".join(str(message.get("content", "")).split())
            for message in reversed(history)
            if str(message.get("role", "")).lower() == "user"
            and str(message.get("content", "")).strip()
        ),
        "",
    )
    if not previous_question or previous_question.casefold() == normalized.casefold():
        return normalized
    return f"{previous_question} Follow-up: {normalized}"


def expand_queries_locally(question: str) -> list[str]:
    """Create conservative keyword variants for Deep mode without extra inference."""
    terms = _content_tokens(question)
    if len(terms) < 2:
        return []
    core = " ".join(dict.fromkeys(terms[:12]))
    variants = [f"{core} details evidence", f"{core} conclusions examples"]
    return [variant for variant in variants if variant.casefold() != question.casefold()]


def is_overview_question(question: str) -> bool:
    query_tokens = set(_content_tokens(question))
    lowered = question.casefold()
    return not query_tokens or any(
        phrase in lowered for phrase in GENERAL_QUESTION_PHRASES
    )


def build_document_profile(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer only high-confidence structural hints from visible document headings."""
    combined = "\n".join(
        normalize_text(str(result.get("text") or "")) for result in results[:20]
    )
    uppercase_text = combined.upper()
    sections = [
        label
        for label, aliases in PROFILE_SECTIONS.items()
        if any(alias in uppercase_text for alias in aliases)
    ]
    section_set = set(sections)
    resume_sections = section_set & {
        "Education",
        "Experience",
        "Skills",
        "Projects",
        "Achievements",
        "Certifications",
    }
    paper_sections = section_set & {
        "Abstract",
        "Introduction",
        "Methodology",
        "Results",
        "Conclusion",
        "References",
    }

    document_type: str | None = None
    if len(resume_sections) >= 3:
        document_type = "résumé/CV"
    elif len(paper_sections) >= 4:
        document_type = "research paper"
    elif (
        "INVOICE" in uppercase_text
        and "TOTAL" in uppercase_text
        and any(term in uppercase_text for term in ("BILL TO", "INVOICE DATE", "DUE DATE"))
    ):
        document_type = "invoice"
    elif (
        "Executive summary" in section_set
        and len(section_set & {"Results", "Recommendations", "Conclusion"}) >= 1
    ):
        document_type = "report"
    elif len(section_set & {"Installation", "Troubleshooting"}) >= 2:
        document_type = "technical manual or guide"

    description = "No reliable document type was detected from the visible headings."
    if document_type:
        description = f"Likely document type: {document_type}."
    if sections:
        description += " Visible sections: " + ", ".join(sections[:8]) + "."
    return {
        "type": document_type,
        "sections": sections[:8],
        "description": description,
    }


def _collect_candidates(
    results: list[dict[str, Any]],
    *,
    general_question: bool,
) -> list[SentenceCandidate]:
    candidates: list[SentenceCandidate] = []
    seen: set[str] = set()
    for source_position, result in enumerate(results, start=1):
        source_relevance = max(0.0, min(float(result.get("relevance") or 0.0), 1.0))
        raw_page = result.get("metadata", {}).get("page")
        page_number = int(raw_page) if str(raw_page).isdigit() else 9999
        fragments = _sentences(normalize_text(str(result.get("text") or "")))
        for sentence_position, fragment in enumerate(fragments):
            normalized = " ".join(fragment.split()).strip(" -*•·▪◦|#\t")
            quality = ocr_text_quality(normalized)
            token_list = _content_tokens(normalized)
            if len(token_list) < 4 or len(normalized) < 32:
                continue
            if quality < (0.46 if general_question else 0.30):
                continue
            if len(normalized.split()) < 6 and normalized[-1:] not in ".!?":
                continue
            if normalized.endswith("?"):
                continue
            if _looks_like_code(normalized) and general_question:
                continue
            if _looks_like_ocr_noise(normalized):
                continue
            if _looks_like_ui_noise(normalized) and general_question:
                continue
            if general_question and not _looks_informative_statement(normalized):
                continue
            if general_question and re.match(
                r"^(?:\*\s*)?(?:here are|here is|here's|the following)\b",
                normalized,
                re.IGNORECASE,
            ):
                continue
            if general_question and _short_token_ratio(normalized) > 0.32:
                continue
            if len(normalized) > 520:
                normalized = normalized[:517].rsplit(" ", 1)[0] + "..."
            dedupe_key = re.sub(r"\W+", " ", normalized.casefold()).strip()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(
                SentenceCandidate(
                    text=normalized,
                    source_id=source_position,
                    source_relevance=source_relevance,
                    source_position=source_position,
                    sentence_position=sentence_position,
                    page_number=page_number,
                    tokens=set(token_list),
                    text_quality=quality,
                    category=_candidate_category(normalized),
                )
            )
    return candidates


def _sentences(text: str) -> list[str]:
    fragments: list[str] = []
    paragraphs = re.split(r"\n\s*\n+", text.replace("[Table]", ""))
    for paragraph in paragraphs:
        reconstructed = re.sub(r"\s*\n\s*", " ", paragraph).strip()
        for fragment in re.split(r"(?<=[.!?])\s+", reconstructed):
            cleaned = " ".join(fragment.split()).strip()
            words = cleaned.split()
            if len(words) <= 72:
                if cleaned:
                    fragments.append(cleaned)
                continue
            for offset in range(0, len(words), 55):
                window = words[offset : offset + 60]
                if window:
                    fragments.append(" ".join(window))
    return fragments


def _score_candidates(
    candidates: list[SentenceCandidate],
    query_tokens: set[str],
    general_question: bool,
) -> None:
    frequency: Counter[str] = Counter()
    for candidate in candidates:
        frequency.update(candidate.tokens)
    maximum_frequency = max(frequency.values(), default=1)

    for candidate in candidates:
        overlap = len(candidate.tokens & query_tokens) / max(len(query_tokens), 1)
        salience = sum(frequency[token] / maximum_frequency for token in candidate.tokens)
        salience /= max(len(candidate.tokens), 1)
        position_bonus = 1 / (1 + candidate.sentence_position)
        source_bonus = 1 / (1 + candidate.source_position * 0.25)
        early_page_bonus = 1 / (1 + max(candidate.page_number - 1, 0) / 4)
        if general_question:
            category_bonus = {
                "education": 0.14,
                "skills": 0.12,
                "experience": 0.10,
                "project": 0.08,
                "achievement": 0.04,
            }.get(candidate.category, 0.0)
            candidate.score = (
                candidate.source_relevance * 0.34
                + candidate.text_quality * 0.24
                + early_page_bonus * 0.14
                + source_bonus * 0.13
                + position_bonus * 0.08
                + salience * 0.07
                + category_bonus
            )
        else:
            candidate.score = (
                overlap * 0.50
                + candidate.source_relevance * 0.23
                + candidate.text_quality * 0.12
                + salience * 0.08
                + position_bonus * 0.04
                + source_bonus * 0.03
            )


def _select_diverse(
    candidates: list[SentenceCandidate],
    limit: int,
) -> list[SentenceCandidate]:
    remaining = sorted(candidates, key=lambda item: item.score, reverse=True)
    selected: list[SentenceCandidate] = []
    while remaining and len(selected) < limit:
        best: SentenceCandidate | None = None
        best_score = -math.inf
        for candidate in remaining[:80]:
            redundancy = max(
                (_jaccard(candidate.tokens, item.tokens) for item in selected),
                default=0.0,
            )
            source_diversity = 0.08 if all(
                item.source_id != candidate.source_id for item in selected
            ) else 0.0
            category_diversity = 0.10 if candidate.category != "other" and all(
                item.category != candidate.category for item in selected
            ) else 0.0
            adjusted = (
                candidate.score
                - redundancy * 0.28
                + source_diversity
                + category_diversity
            )
            if adjusted > best_score:
                best = candidate
                best_score = adjusted
        if best is None:
            break
        remaining.remove(best)
        if selected and max(_jaccard(best.tokens, item.tokens) for item in selected) > 0.84:
            continue
        selected.append(best)
    return selected


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if token not in STOP_WORDS and len(token) > 1]


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _looks_like_code(text: str) -> bool:
    code_markers = sum(text.count(marker) for marker in ("{", "}", ";", "#include", "<<", ">>"))
    return code_markers >= 3


def _looks_like_ocr_noise(text: str) -> bool:
    if re.search(r"(?:[-_=<>~—–]{4,}|[\[\]{}|<>~=]{3,})", text):
        return True
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 6:
        return False
    isolated_fragments = sum(
        len(word) <= 2 and word not in {"a", "an", "as", "at", "be", "by", "in", "is", "it", "of", "on", "or", "to"}
        for word in words
    )
    return isolated_fragments / len(words) > 0.24


def _candidate_category(text: str) -> str:
    lowered = text.casefold().lstrip("•·▪◦-* ")
    if any(
        term in lowered
        for term in ("education ", "b.tech", "bachelor", "master of", "university", "college")
    ):
        return "education"
    if any(
        term in lowered
        for term in (
            "skills ",
            "skills:",
            "languages:",
            "frameworks:",
            "databases:",
            "technologies:",
            "coursework",
        )
    ):
        return "skills"
    if lowered.startswith(("rated ", "earned ", "qualified ", "solved ", "awarded ")):
        return "achievement"
    if "work experience" in lowered or "professional experience" in lowered:
        return "experience"
    if lowered.startswith(("built ", "developed ", "implemented ", "designed ")):
        return "project"
    if lowered.startswith(tuple(f"{verb} " for verb in ACTION_STARTERS)):
        return "experience"
    return "other"


def _looks_like_ui_noise(text: str) -> bool:
    lowered = f" {text.casefold()} "
    markers = (
        " explorer ",
        " outline ",
        " timeline ",
        " spaces:",
        " utf-8",
        " go live",
        " prettier ",
        " node_modules ",
        " package-lock.json",
    )
    return sum(marker in lowered for marker in markers) >= 2


def _looks_informative_statement(text: str) -> bool:
    lowered = f" {text.casefold()} "
    signals = (
        " is ",
        " are ",
        " means ",
        " stands for ",
        " refers to ",
        " provides ",
        " includes ",
        " introduces ",
        " covers ",
        " discusses ",
        " focuses on ",
        " explains ",
        " describes ",
        " enables ",
        " allows ",
        " supports ",
        " uses ",
        " defines ",
        " contains ",
        " consists of ",
        " is used ",
        " are used ",
    )
    tokens = _tokens(text.lstrip("•·▪◦-* "))
    action_statement = bool(tokens and tokens[0] in ACTION_STARTERS)
    structured_statement = any(
        lowered.strip().startswith(prefix)
        for prefix in (
            "education ",
            "languages:",
            "frameworks:",
            "databases:",
            "skills:",
            "skills ",
            "projects ",
            "work experience ",
            "professional experience ",
            "achievements ",
            "technologies:",
        )
    )
    return any(signal in lowered for signal in signals) or action_statement or structured_statement


def _short_token_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 1.0
    return sum(len(word) <= 2 for word in words) / len(words)


def _document_title(results: list[dict[str, Any]]) -> str:
    filenames = [
        str(result.get("metadata", {}).get("filename") or "").strip()
        for result in results
    ]
    unique = [filename for filename in dict.fromkeys(filenames) if filename]
    if len(unique) != 1:
        return ""
    title = re.sub(r"\.pdf$", "", unique[0], flags=re.IGNORECASE)
    title = re.sub(r"[_-]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    if not title or re.fullmatch(r"[0-9a-f]{20,}", title, re.IGNORECASE):
        return ""
    return title
