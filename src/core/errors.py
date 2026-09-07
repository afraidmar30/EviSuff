"""Shared runtime exceptions."""

from __future__ import annotations


class ApiQuotaExceeded(RuntimeError):
    """Raised when an external API quota/rate/billing/auth limit is hit.

    Batch runners should treat this as a fatal, resumable stop: keep already
    written results, do not mark the current sample complete, and exit.
    """

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class ExternalServiceFailure(RuntimeError):
    """Raised when a required external service fails and the sample must retry.

    Unlike normal tool-level soft failures, batch runners should stop without
    writing a result row so resume can retry the current sample cleanly.
    """

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"{provider}: {message}")


def looks_like_quota_error(status_code: int | None, text: str = "") -> bool:
    """Return True for quota/rate/billing/auth failures that should stop a run."""
    normalized = text.lower()
    if status_code in {401, 402, 403, 429}:
        return True
    quota_markers = (
        "quota",
        "rate limit",
        "ratelimit",
        "too many requests",
        "insufficient",
        "billing",
        "balance",
        "credit",
        "credits",
        "exceeded",
        "exhausted",
        "unauthorized",
        "forbidden",
        "payment required",
    )
    return any(marker in normalized for marker in quota_markers)


def quota_error_from_exception(provider: str, error: Exception) -> ApiQuotaExceeded | None:
    """Convert quota/auth/rate-limit-like exceptions into a fatal run stop."""
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    body = ""
    if response is not None:
        if status_code is None:
            status_code = getattr(response, "status_code", None)
        body = str(getattr(response, "text", "") or "")[:500]
    text = f"{type(error).__name__}: {error}"
    if body:
        text = f"{text}\n{body}"
    if looks_like_quota_error(status_code, text):
        status = f"HTTP {status_code}: " if status_code is not None else ""
        return ApiQuotaExceeded(provider, f"{status}{text[:1000]}")
    return None
