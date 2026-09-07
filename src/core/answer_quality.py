"""Lightweight final-answer quality checks for agentic runs."""

from __future__ import annotations

import re


DEFAULT_MIN_FINAL_ANSWER_CHARS = 200
DEFAULT_SFT_MIN_FINAL_ANSWER_CHARS = 800

_PROCEDURAL_PREFIXES = (
    "let me ",
    "i will ",
    "i'll ",
    "i need to ",
    "i should ",
    "i'm going to ",
    "now i will ",
    "next, i will ",
    "checking ",
    "searching ",
)

_PROCEDURAL_PATTERNS = (
    re.compile(r"\blet me (?:check|fetch|search|look|read|try|open)\b", re.I),
    re.compile(r"\bi(?:'ll| will) (?:check|fetch|search|look|read|try|open)\b", re.I),
    re.compile(r"\b(?:need|needs) to (?:check|fetch|search|look|verify)\b", re.I),
)


def final_answer_quality_issue(answer: str | None, *, min_chars: int = DEFAULT_MIN_FINAL_ANSWER_CHARS) -> str | None:
    """Return a rejection reason when text is clearly not a usable final answer."""
    text = (answer or "").strip()
    if not text:
        return "empty_final_answer"
    if len(text) < min_chars:
        return "final_answer_too_short"
    if looks_like_procedural_fragment(text):
        return "procedural_fragment_final_answer"
    return None


def looks_like_procedural_fragment(answer: str | None) -> bool:
    """Detect agent scratchpad/status text accidentally captured as final answer."""
    text = (answer or "").strip()
    if not text:
        return False
    lowered = text.lower()
    first_line = lowered.splitlines()[0].strip()
    if any(first_line.startswith(prefix) for prefix in _PROCEDURAL_PREFIXES):
        return True
    if len(text) <= 500 and any(pattern.search(text) for pattern in _PROCEDURAL_PATTERNS):
        return True
    # Short texts with no sentence-level conclusion are usually unfinished agent chatter.
    if len(text) <= 500 and not re.search(r"\b(?:answer|conclusion|therefore|insufficient|sufficient|because|evidence|sources?)\b", lowered):
        return True
    return False
