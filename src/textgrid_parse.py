"""Parse PriMock57's Praat "long" TextGrid transcripts (one IntervalTier per
speaker channel) into a plain list of {start, end, text} utterances.

PriMock57 ships one TextGrid per (consultation, speaker) pair, e.g.
`transcripts/day1_consultation01_doctor.TextGrid`. Unlike the bridge-domain
CLAIP pipeline (pyannote speaker diarization + faster-whisper ASR), speaker
separation and utterance timing here are already human-annotated ground
truth, so no diarization model is needed (see
docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 3, Decisions).

Tags used by the transcribers (see transcripts/README.md):
  - `<UNSURE>...</UNSURE>`: transcriber was unsure; keep the inner text.
  - `<UNIN/>`: an unintelligible audio span; drop the marker, keep the rest.

Usage: imported, not run directly.
"""

from __future__ import annotations

import re
from pathlib import Path

_INTERVAL_PATTERN = re.compile(
    r'xmin\s*=\s*([\d.eE+-]+)\s*\r?\n\s*xmax\s*=\s*([\d.eE+-]+)\s*\r?\n\s*text\s*=\s*"((?:[^"]|"")*)"',
    re.MULTILINE,
)
_UNSURE_PATTERN = re.compile(r"<UNSURE>(.*?)</UNSURE>", re.DOTALL)
_UNIN_PATTERN = re.compile(r"<UNIN/>")


def _clean_text(raw: str) -> str:
    """Strip PriMock57 transcriber tags, keeping <UNSURE> inner text and
    dropping <UNIN/> markers, then collapse whitespace."""
    text = raw.replace('""', '"')  # TextGrid escapes embedded quotes as ""
    text = _UNSURE_PATTERN.sub(lambda m: m.group(1), text)
    text = _UNIN_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_textgrid(path: Path | str) -> list[dict]:
    """Parse a single Praat long-format TextGrid file (one IntervalTier) into
    a time-sorted list of {start, end, text}. Intervals whose text is empty
    after tag-cleaning (silence, or an interval that was only `<UNIN/>`) are
    dropped."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")

    results: list[dict] = []
    for xmin_str, xmax_str, text_raw in _INTERVAL_PATTERN.findall(content):
        text = _clean_text(text_raw)
        if not text:
            continue
        results.append({"start": float(xmin_str), "end": float(xmax_str), "text": text})

    results.sort(key=lambda s: s["start"])
    return results
