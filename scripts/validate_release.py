#!/usr/bin/env python3
"""Fail on common accidental disclosures in the public code tree."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "__pycache__"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".sh", ".html", ".js", ".css", ".jinja"}
FORBIDDEN_PATHS = re.compile(r"/(?:data\d*|home|root|mnt)/[A-Za-z0-9_.-]+/")
LIVE_SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|jina_[A-Za-z0-9]{20,})"
)
PRIVATE_IP = re.compile(r"https?://(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)")


def main() -> None:
    issues: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        rel = path.relative_to(ROOT)
        if path.name == ".env":
            issues.append(f"forbidden environment file: {rel}")
        if path.stat().st_size > 50 * 1024 * 1024:
            issues.append(f"unexpected file larger than 50 MiB: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".env.example", ".gitignore"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if FORBIDDEN_PATHS.search(text):
            issues.append(f"absolute private path: {rel}")
        if PRIVATE_IP.search(text):
            issues.append(f"private network endpoint: {rel}")
        if LIVE_SECRET.search(text):
            issues.append(f"possible live credential: {rel}")
    if issues:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(sorted(set(issues))))
    print("release validation passed")


if __name__ == "__main__":
    main()
