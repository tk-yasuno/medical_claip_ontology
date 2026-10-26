"""Phase 3, Step 6: merge a consultation's doctor + patient TextGrids into a
single time-sorted, speaker-tagged segment list -- the medical-domain
equivalent of the bridge CLAIP pipeline's diarization.json, but built from
PriMock57's human-annotated utterance transcripts instead of a pyannote
diarization model (see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 3,
Decisions: speaker channels are already separated and hand-transcribed, so
no diarization model is needed here).

Usage:
    python src/merge_speaker_segments.py --consultation day1_consultation01
    python src/merge_speaker_segments.py --all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from textgrid_parse import parse_textgrid

DEFAULT_TRANSCRIPTS_DIR = Path("data/primock57/transcripts")
DEFAULT_AUDIO_DIR = Path("data/primock57/audio")
DEFAULT_OUTPUT_ROOT = Path("data/processed/audio_text")

_CONSULTATION_ID_PATTERN = re.compile(r"^(day\d+_consultation\d+)_(doctor|patient)\.TextGrid$")


def list_consultation_ids(transcripts_dir: Path | str = DEFAULT_TRANSCRIPTS_DIR) -> list[str]:
    """Every `day{D}_consultation{NN}` stem that has both a doctor and a
    patient TextGrid, sorted for reproducible iteration order."""
    transcripts_dir = Path(transcripts_dir)
    ids: set[str] = set()
    found: dict[str, set[str]] = {}
    for path in transcripts_dir.glob("*.TextGrid"):
        match = _CONSULTATION_ID_PATTERN.match(path.name)
        if not match:
            continue
        found.setdefault(match.group(1), set()).add(match.group(2))
    for consultation_id, speakers in found.items():
        if {"doctor", "patient"} <= speakers:
            ids.add(consultation_id)
    return sorted(ids)


def merge_consultation(
    consultation_id: str,
    transcripts_dir: Path | str = DEFAULT_TRANSCRIPTS_DIR,
    audio_dir: Path | str = DEFAULT_AUDIO_DIR,
) -> list[dict]:
    """Parse both channels of one consultation and merge into a single
    time-sorted list of {start, end, speaker, text_gold, wav_path}."""
    transcripts_dir = Path(transcripts_dir)
    audio_dir = Path(audio_dir)

    segments: list[dict] = []
    for speaker in ("doctor", "patient"):
        tg_path = transcripts_dir / f"{consultation_id}_{speaker}.TextGrid"
        wav_path = audio_dir / f"{consultation_id}_{speaker}.wav"
        for seg in parse_textgrid(tg_path):
            segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": speaker,
                    "text_gold": seg["text"],
                    "wav_path": str(wav_path),
                }
            )
    segments.sort(key=lambda s: s["start"])
    return segments


def save_segments(segments: list[dict], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge doctor+patient TextGrids into segments.json")
    parser.add_argument("--consultation", help="Single consultation id, e.g. day1_consultation01")
    parser.add_argument("--all", action="store_true", help="Process every consultation found in transcripts/")
    parser.add_argument("--transcripts-dir", default=str(DEFAULT_TRANSCRIPTS_DIR))
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    if args.all:
        consultation_ids = list_consultation_ids(args.transcripts_dir)
    elif args.consultation:
        consultation_ids = [args.consultation]
    else:
        parser.error("specify --consultation <id> or --all")
        return

    total_segments = 0
    for consultation_id in consultation_ids:
        segments = merge_consultation(consultation_id, args.transcripts_dir, args.audio_dir)
        out_path = Path(args.output_root) / consultation_id / "segments.json"
        save_segments(segments, out_path)
        n_doctor = sum(1 for s in segments if s["speaker"] == "doctor")
        n_patient = len(segments) - n_doctor
        print(f"{consultation_id}: {len(segments)} segments ({n_doctor} doctor, {n_patient} patient) -> {out_path}")
        total_segments += len(segments)

    print(f"OK: {len(consultation_ids)} consultations, {total_segments} segments total")


if __name__ == "__main__":
    main()
