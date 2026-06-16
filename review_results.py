#!/usr/bin/env python3
"""Persistence of FP-review verdicts.

A single `fp_review_results.json` per folder maps each clip filename to its
verdict, enabling resume across sessions and an audit trail. Updates follow an
immutable pattern (a new dict is returned, never mutated in place).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_FILENAME = "fp_review_results.json"
VERDICT_TP = "TP"
VERDICT_FP = "FP"
_VALID_VERDICTS = (VERDICT_TP, VERDICT_FP)


def results_path_for(folder: Path) -> Path:
    return folder / RESULTS_FILENAME


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing verdicts, dropping any malformed entries. Never raises."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict):
        return {}
    clean: dict[str, dict[str, Any]] = {}
    for name, entry in results.items():
        if isinstance(entry, dict) and entry.get("verdict") in _VALID_VERDICTS:
            clean[str(name)] = entry
    return clean


def record_verdict(
    results: dict[str, dict[str, Any]],
    clip_name: str,
    verdict: str,
    annotation_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a new results dict with `clip_name` set to the given verdict."""
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {verdict!r}")
    entry = {
        "verdict": verdict,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotation": annotation_name,
    }
    return {**results, clip_name: entry}


def save_results(
    path: Path,
    folder: Path,
    results: dict[str, dict[str, Any]],
) -> None:
    """Atomically write results (temp file + replace). Raises OSError on failure."""
    payload = {"folder": str(folder), "results": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
