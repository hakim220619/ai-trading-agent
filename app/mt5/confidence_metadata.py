"""Encode confidence metadata in MT5 order comments."""
from __future__ import annotations

import re


_CONFIDENCE_PATTERN = re.compile(r"(?:^|-)cf-(\d{1,5})(?:-|$)", re.IGNORECASE)
_CYCLE_PATTERN = re.compile(r"^m1r-([a-z0-9]{6})-", re.IGNORECASE)
_COMPACT_RECOVERY_PATTERN = re.compile(r"^r([a-z0-9]{4})s\d+[bs]c(\d{3})", re.IGNORECASE)


def confidence_comment(confidence: float, direction: str) -> str:
    """Return a compact MT5 comment with confidence stored in basis points."""
    basis_points = round(max(0.0, min(1.0, float(confidence))) * 10_000)
    return f"ai-cf-{basis_points:05d}-{direction.lower()}"


def parse_confidence_pct(comment: object) -> float | None:
    """Read a percentage from a comment created by ``confidence_comment``."""
    match = _CONFIDENCE_PATTERN.search(str(comment or ""))
    if match:
        return round(int(match.group(1)) / 100.0, 2)
    compact = _COMPACT_RECOVERY_PATTERN.search(str(comment or ""))
    return round(int(compact.group(2)) / 10.0, 2) if compact else None


def parse_cycle_key(comment: object) -> str | None:
    match = _CYCLE_PATTERN.search(str(comment or ""))
    if match:
        return match.group(1).upper()
    compact = _COMPACT_RECOVERY_PATTERN.search(str(comment or ""))
    return compact.group(1).upper() if compact else None
