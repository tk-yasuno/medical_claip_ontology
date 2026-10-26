"""Phase 7, Step 17: build (audio, text) triples for the Audio-Text CLAIP
branch from PriMock57's ASR'd utterance segments and consultation notes.

Medical-domain analogue of the bridge CLAIP pipeline's
src_claip/build_triples.py segment-centric mode, but inverted: there, many
(dense, segment-aligned) *frames* could map to one segment's audio+text;
here, *all utterance audio segments of one consultation* map to that single
consultation's one note text (PriMock57 provides one clinician-written note
per encounter, not per-utterance). So each consultation contributes
len(segments) triples, all sharing the same positive text embedding -- the
in-batch-negative InfoNCE loss (train_projection_pair.py) still works
because other consultations' audio is paired with a *different* note.

Usage:
    python src_claip/build_triples_audio_text.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from encoders_medical import AudioEncoder, TextEncoder

DEFAULT_PROCESSED_ROOT = Path("data/processed/audio_text")
DEFAULT_NOTES_DIR = Path("data/primock57/notes")
DEFAULT_ASR_TAG = "dictonly"  # see docs: re-decoding with initial_prompt (tag "corrected") empirically
                              # increased WER (more empty hypotheses) for this MVP's small 5-term
                              # vocabulary; "dictonly" (domain-dict postprocess on the raw ASR output,
                              # no re-decoding) is the adopted, non-regressive correction strategy.
                              # Falls back to "raw" if absent, see main().


def load_note_text(notes_dir: Path, consultation_id: str) -> str:
    note_path = notes_dir / f"{consultation_id}.json"
    with open(note_path, encoding="utf-8") as f:
        note = json.load(f)
    # presenting_complaint + note body: the full text searchable by a
    # clinician's free-text query (see eval_cross_domain_retrieval.py).
    return f"{note.get('presenting_complaint', '')}. {note.get('note', '')}".strip()


def build_triples(
    processed_root: Path,
    notes_dir: Path,
    consultation_ids: list[str],
    asr_tag: str,
    audio_encoder: AudioEncoder,
    text_encoder: TextEncoder,
) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    triples_meta: list[dict] = []
    audio_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []
    text_emb_cache: dict[str, np.ndarray] = {}

    for consultation_id in consultation_ids:
        work_dir = processed_root / consultation_id
        asr_path = work_dir / f"asr_{asr_tag}.json"
        if not asr_path.is_file():
            asr_path = work_dir / "asr_raw.json"
        if asr_path.is_file():
            with open(asr_path, encoding="utf-8") as f:
                segments = json.load(f)
            has_asr = True
        else:
            # ASR (asr_medical.py) was only run on a 30-consultation subset
            # (see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Decisions:
            # large-v3 decode time), but CLAIP's audio embedding needs only
            # the raw waveform (AudioEncoder.encode_audio_segment), not an
            # ASR transcript -- so fall back to segments.json (available
            # for all 57 consultations, see merge_speaker_segments.py) to
            # extend the Audio-Text branch's audio+note coverage beyond the
            # ASR/WER experiment's 30-consultation scope (v0.1.1 fix: a
            # 30-consultation text corpus was too small a vocabulary
            # surface for eval_cross_domain_retrieval.py's shared-query
            # selection to find many genuinely co-occurring terms).
            segments_path = work_dir / "segments.json"
            if not segments_path.is_file():
                print(f"  skip {consultation_id}: no asr_*.json or segments.json found")
                continue
            with open(segments_path, encoding="utf-8") as f:
                segments = json.load(f)
            has_asr = False

        note_text = load_note_text(notes_dir, consultation_id)
        if consultation_id not in text_emb_cache:
            text_emb_cache[consultation_id] = text_encoder.encode_text(note_text)
        text_emb = text_emb_cache[consultation_id]

        for seg in segments:
            audio_emb = audio_encoder.encode_audio_segment(seg["wav_path"], seg["start"], seg["end"])
            triples_meta.append(
                {
                    "consultation_id": consultation_id,
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text_asr": (seg.get("text_corrected") or seg.get("text_asr", "")) if has_asr else "",
                    "has_asr": has_asr,
                    "note_text": note_text,
                    "row": len(triples_meta),
                }
            )
            audio_embs.append(audio_emb)
            text_embs.append(text_emb)

    return triples_meta, audio_embs, text_embs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audio/text embedding triples for the Audio-Text CLAIP branch")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--notes-dir", default=str(DEFAULT_NOTES_DIR))
    parser.add_argument("--asr-tag", default=DEFAULT_ASR_TAG)
    parser.add_argument("--output-dir-name", default="claip")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--consultation", action="append", default=None)
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    if args.all:
        consultation_ids = sorted(p.name for p in processed_root.iterdir() if (p / "segments.json").is_file())
    elif args.consultation:
        consultation_ids = args.consultation
    else:
        parser.error("specify --consultation <id> (repeatable) or --all")
        return

    print("loading encoders (audio/text)...")
    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()

    triples_meta, audio_embs, text_embs = build_triples(
        processed_root, Path(args.notes_dir), consultation_ids, args.asr_tag, audio_encoder, text_encoder
    )

    audio_encoder.unload()
    text_encoder.unload()

    audio_embs = np.stack(audio_embs) if audio_embs else np.zeros((0, 1), dtype="float32")
    text_embs = np.stack(text_embs) if text_embs else np.zeros((0, 1), dtype="float32")

    out_dir = processed_root / args.output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "audio_emb.npy", audio_embs)
    np.save(out_dir / "text_emb.npy", text_embs)
    with open(out_dir / "triples.json", "w", encoding="utf-8") as f:
        json.dump(triples_meta, f, ensure_ascii=False, indent=2)

    print(
        f"OK: {len(triples_meta)} triples from {len(consultation_ids)} consultations -> {out_dir} "
        f"(audio_emb {audio_embs.shape}, text_emb {text_embs.shape})"
    )


if __name__ == "__main__":
    main()
