"""Dataset loader for RAES-Bench releases.

The loader keeps gold fields available to evaluators while exposing a
minimal public view for agents. Runner code should pass only
``entry.public_view()`` or ``entry.question`` into agents.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from src.raes_eval.schemas import GOLD_FIELDS, RAESTask


PUBLIC_FIELDS = ("id", "category", "domain", "task_type", "difficulty", "question", "split")
REQUIRED_PUBLIC_FIELDS = ("id", "question")


@dataclass(frozen=True)
class RAESEntry:
    """One RAES item, with helpers for gold-hidden evaluation."""

    raw: dict

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def question(self) -> str:
        return str(self.raw["question"])

    @property
    def category(self) -> str:
        return str(self.raw.get("category", ""))

    @property
    def domain(self) -> str:
        return str(self.raw.get("domain", self.category))

    @property
    def task_type(self) -> str:
        return str(self.raw.get("task_type", ""))

    @property
    def difficulty(self) -> str:
        return str(self.raw.get("difficulty", ""))

    @property
    def answerability(self) -> str:
        return str(self.raw.get("answerability", ""))

    @property
    def split(self) -> str:
        return str(self.raw.get("split", ""))

    @property
    def has_gold(self) -> bool:
        return "gold_answer" in self.raw and "gold_sources" in self.raw

    def public_view(self) -> dict:
        """Return the fields an agent is allowed to see."""
        return {field: self.raw[field] for field in PUBLIC_FIELDS if field in self.raw}

    def to_task(self, mode: Literal["agent", "gold"] = "agent") -> RAESTask:
        """Return the standardized RAES task object."""
        return RAESTask.from_mapping(self.raw, mode=mode)

    def required_facets(self) -> list[str]:
        facets = self.raw.get("required_facets") or {}
        if isinstance(facets, list):
            return [str(facet) for facet in facets if facet]
        if isinstance(facets, dict):
            out: list[str] = []
            for key in ("content_facets", "evidence_facets", "reasoning_facets"):
                out.extend(str(facet) for facet in facets.get(key, []) if facet)
            return out
        return []

    def gold_source_urls(self) -> list[str]:
        return [
            str(source.get("url"))
            for source in self.raw.get("gold_sources", []) or []
            if source.get("url")
        ]


class RAESDataset:
    """Load and filter a RAES JSONL file."""

    def __init__(
        self,
        path: str | Path,
        split: str | None = None,
        limit: int | None = None,
        sample: int | None = None,
        seed: int = 20260519,
    ):
        self.path = Path(path)
        self.entries = [RAESEntry(row) for row in self._load_jsonl(self.path)]
        self._validate()
        if split:
            self.entries = [entry for entry in self.entries if entry.split == split]
        if sample is not None:
            rng = random.Random(seed)
            rows = self.entries[:]
            rng.shuffle(rows)
            self.entries = rows[:sample]
        if limit is not None:
            self.entries = self.entries[:limit]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterable[RAESEntry]:
        return iter(self.entries)

    def by_id(self) -> dict[str, RAESEntry]:
        return {entry.id: entry for entry in self.entries}

    def public_items(self) -> list[dict]:
        return [entry.public_view() for entry in self.entries]

    def assert_no_gold_leakage(self) -> None:
        for entry in self.entries:
            leaked = GOLD_FIELDS.intersection(entry.public_view())
            if leaked:
                raise ValueError(f"{entry.id} public view leaks gold fields: {sorted(leaked)}")

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        return rows

    def _validate(self) -> None:
        seen = set()
        for entry in self.entries:
            missing = [field for field in REQUIRED_PUBLIC_FIELDS if field not in entry.raw]
            if missing:
                raise ValueError(f"entry missing required fields {missing}: {entry.raw}")
            if entry.id in seen:
                raise ValueError(f"duplicate RAES id: {entry.id}")
            seen.add(entry.id)


def load_raes_tasks(
    path: str | Path,
    mode: Literal["agent", "gold"] = "agent",
) -> list[RAESTask]:
    """Load RAES tasks in agent-safe or gold-aware mode."""
    rows = RAESDataset._load_jsonl(Path(path))
    return [RAESTask.from_mapping(row, mode=mode) for row in rows]
