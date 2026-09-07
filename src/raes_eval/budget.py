"""Budget definitions for RAES benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    max_turns: int = 20
    max_search: int = 10
    max_fetch: int = 10
    academic_search: int = 5
    news_search: int = 5

    @classmethod
    def preset(cls, name: str) -> "Budget":
        normalized = name.lower()
        if normalized == "low":
            return cls(max_turns=20, max_search=10, max_fetch=10, academic_search=5, news_search=5)
        if normalized == "medium":
            return cls(max_turns=40, max_search=25, max_fetch=25, academic_search=10, news_search=10)
        if normalized == "high":
            return cls(max_turns=80, max_search=50, max_fetch=50, academic_search=20, news_search=20)
        raise ValueError(f"unknown budget preset: {name}")

    @classmethod
    def from_config(cls, config: dict | None) -> "Budget":
        cfg = config or {}
        if "budget" in cfg and isinstance(cfg["budget"], str):
            base = cls.preset(cfg["budget"])
        else:
            base = cls()
        budgets = cfg.get("budgets", {}) or {}
        return cls(
            max_turns=int(cfg.get("max_turns", base.max_turns)),
            max_search=int(budgets.get("web_search", cfg.get("max_search", base.max_search))),
            max_fetch=int(budgets.get("web_fetch", cfg.get("max_fetch", base.max_fetch))),
            academic_search=int(budgets.get("academic_search", base.academic_search)),
            news_search=int(budgets.get("news_search", base.news_search)),
        )

    def to_dict(self) -> dict:
        return {
            "max_turns": self.max_turns,
            "max_search": self.max_search,
            "max_fetch": self.max_fetch,
            "academic_search": self.academic_search,
            "news_search": self.news_search,
        }
