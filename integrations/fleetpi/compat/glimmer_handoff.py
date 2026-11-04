"""Keep Glimmer's reasoning/tool phase transition on the delivered-text clock."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterable, Sequence
from pathlib import Path

from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser


# This fix relies on extract_reasoning_streaming running immediately before the
# predicate in DelegatingParser.parse_delta, and on _tool_handoff_done becoming
# true only when a complete framed handoff is returned. Refuse unreviewed drift.
EXPECTED_SOURCES = {
    "reasoning/muse_glimmer_reasoning_parser.py":
        "94b0ae907018a484fa9aab8167fb32ec6d5e390268ca51e3d60868df5868e90b",
    "parser/abstract_parser.py":
        "35810e909e8927fa087d5dfbdaa85bb3d2920faa51dae6b6780c8b1a915f4368",
    "tool_parsers/muse_glimmer_tool_parser.py":
        "0159565ce6634cddb8d8dcd72732cef153f6ec0920f2a2e6dd553eb25301581a",
}


def verify_sources(vllm_root: Path) -> None:
    """Reject a parser combination whose transition contract was not replayed."""
    for relative, expected in EXPECTED_SOURCES.items():
        actual = hashlib.sha256((vllm_root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"FleetPi Glimmer compatibility source mismatch: {relative}: "
                f"expected {expected}, got {actual}; review and replay before updating"
            )


verify_sources(Path(inspect.getfile(MuseGlimmerReasoningParser)).resolve().parents[1])


@ReasoningParserManager.register_module("fleetpi_glimmer_handoff")
class FleetPiGlimmerReasoningParser(MuseGlimmerReasoningParser):
    """Transition only when this same delta supplied its intact tool header."""

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # Raw token IDs may lead detokenized text (e.g. stop-string holdback).
        # The inherited extractor just set this flag iff it returned the full
        # tool-channel prefix as .content. A token-based decision can instead
        # make DelegatingParser replace current_text with an empty handoff.
        return self._tool_handoff_done
