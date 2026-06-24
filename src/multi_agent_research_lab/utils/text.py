"""Small text helpers shared by agents and evaluation."""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[(\d+)\]")


def count_citation_markers(text: str, max_id: int) -> int:
    """Count distinct inline ``[n]`` citation markers in ``text`` where 1 <= n <= max_id.

    Used to measure how many of the available sources the LLM actually cited.
    """

    if max_id <= 0:
        return 0
    ids = {int(m) for m in _CITATION_RE.findall(text)}
    return len({n for n in ids if 1 <= n <= max_id})
