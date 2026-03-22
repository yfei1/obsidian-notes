"""
engine.overlap — Detect content overlap between notes for dedup strategy.

Wraps shared.find_paragraph_overlaps to return typed Overlap objects
for use by the evolution loop's strategy selection.
"""

from dataclasses import dataclass

from shared import find_paragraph_overlaps


@dataclass
class Overlap:
    """A detected content overlap between two notes."""
    source_path: str           # note with the duplicate content
    canonical_path: str        # note that should be the canonical home
    source_preview: str        # preview of the overlapping paragraph in source
    canonical_preview: str     # preview of the matching content in canonical
    overlap_ratio: float       # 0-1, how much overlap


def detect_overlaps(
    file_contents: dict[str, str],
    threshold: float = 0.7,
    min_paragraph_len: int = 100,
) -> list[Overlap]:
    """Detect paragraph-level content overlaps across all notes.

    Returns a list of Overlap objects sorted by overlap ratio descending.
    """
    overlaps = []
    paths = sorted(file_contents.keys())

    for i, source_path in enumerate(paths):
        # Only compare against notes that come after (avoid double-counting)
        target_contents = {p: file_contents[p] for p in paths[i + 1:]}
        raw = find_paragraph_overlaps(
            file_contents[source_path], target_contents,
            threshold=threshold, min_paragraph_len=min_paragraph_len,
        )
        for target_path, ratio, src_preview, tgt_preview in raw:
            overlaps.append(Overlap(
                source_path=source_path,
                canonical_path=target_path,
                source_preview=src_preview,
                canonical_preview=tgt_preview,
                overlap_ratio=ratio,
            ))

    overlaps.sort(key=lambda o: o.overlap_ratio, reverse=True)
    return overlaps
