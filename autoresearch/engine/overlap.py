"""
engine.overlap — Detect content overlap between notes for dedup strategy.

Reuses the word-set overlap approach from legacy score.py's score_uniqueness,
but returns actionable data (which paragraphs overlap with which notes).
"""

import re
from dataclasses import dataclass
from pathlib import Path


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

    For each note, checks if any substantial paragraph overlaps with content
    in any other note. Returns a list of Overlap objects sorted by overlap ratio.

    Args:
        file_contents: {relative_path: content} for all notes.
        threshold: minimum word-set overlap ratio to flag (0-1).
        min_paragraph_len: minimum paragraph length in chars to consider.

    Returns:
        List of Overlap objects, sorted by overlap_ratio descending.
    """
    overlaps = []
    paths = sorted(file_contents.keys())

    for i, source_path in enumerate(paths):
        source_content = file_contents[source_path]
        paragraphs = re.split(r'\n\s*\n', source_content)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > min_paragraph_len]

        for canonical_path in paths[i + 1:]:
            canonical_content = file_contents[canonical_path]
            canonical_sentences = re.split(r'[.!?]\s+', canonical_content)

            for para in paragraphs:
                para_words = set(para.lower().split())
                if len(para_words) < 10:
                    continue

                for j in range(len(canonical_sentences) - 2):
                    chunk = ' '.join(canonical_sentences[j:j + 3])
                    chunk_words = set(chunk.lower().split())
                    if len(chunk_words) < 10:
                        continue

                    common = len(para_words & chunk_words)
                    ratio = common / min(len(para_words), len(chunk_words))

                    if ratio > threshold:
                        overlaps.append(Overlap(
                            source_path=source_path,
                            canonical_path=canonical_path,
                            source_preview=para.replace('\n', ' ')[:150],
                            canonical_preview=chunk.replace('\n', ' ')[:150],
                            overlap_ratio=ratio,
                        ))
                        break  # one overlap per note pair is enough

    overlaps.sort(key=lambda o: o.overlap_ratio, reverse=True)
    return overlaps
