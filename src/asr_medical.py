"""Medical-domain ASR stage (Phase 3, Step 7): transcribe each PriMock57
utterance segment with faster-whisper (English), the medical-domain
counterpart of the bridge CLAIP pipeline's src/asr.py.

Differences from the bridge-domain asr.py (see
docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 3 / Decisions):
  - `language="en"` (PriMock57 is UK English), not "ja".
  - No diarization.json input: segments.json (merge_speaker_segments.py)
    already carries {start, end, speaker, text_gold, wav_path}, with a
    separate wav per speaker channel, so each segment is cut directly from
    its own channel's wav file rather than from one shared meeting wav.
  - `text_gold` (PriMock57's human transcript) is carried through
    unchanged in the output, for wer_eval_medical.py to use as reference.

Usage:
    python src/asr_medical.py --consultation day1_consultation01
    python src/asr_medical.py --all
    python src/asr_medical.py --all --vocab domain_dict/vocab_medical.yaml --tag corrected
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
import yaml
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_PROCESSED_ROOT = Path("data/processed/audio_text")

_model_cache: WhisperModel | None = None
_model_cache_key: tuple[str, str, str] | None = None


def load_model(model_size: str = DEFAULT_MODEL_SIZE, device: str | None = None, compute_type: str | None = None) -> WhisperModel:
    """Load (and cache) faster-whisper. See src/asr.py (bridge domain) for
    the matching device/compute_type fallback rationale."""
    global _model_cache, _model_cache_key
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    cache_key = (model_size, device, compute_type)
    if _model_cache is not None and _model_cache_key == cache_key:
        return _model_cache
    _model_cache = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_cache_key = cache_key
    return _model_cache


def unload_model() -> None:
    global _model_cache, _model_cache_key
    if _model_cache is not None:
        del _model_cache
        _model_cache = None
        _model_cache_key = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_wav_cache: dict[str, np.ndarray] = {}


def load_wav_mono16k(path: Path | str) -> np.ndarray:
    path = str(path)
    if path in _wav_cache:
        return _wav_cache[path]
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
            raise ValueError(f"{path} is not 16kHz mono (got {wf.getframerate()}Hz, {wf.getnchannels()}ch)")
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    _wav_cache[path] = audio
    return audio


def build_initial_prompt(vocab_path: Path | str | None, max_terms: int = 40) -> str:
    """Build a short English initial_prompt from vocab_medical.yaml's
    canonical term list (see src/asr.py's Japanese equivalent). Returns ""
    if vocab_path is None/missing, matching the "baseline, no domain dict"
    run used for the pre-correction WER measurement
    (docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 3/6)."""
    if not vocab_path:
        return ""
    vocab_path = Path(vocab_path)
    if not vocab_path.is_file():
        return ""
    with open(vocab_path, encoding="utf-8") as f:
        vocab = yaml.safe_load(f) or {}
    terms = list(vocab.keys())[:max_terms]
    return ", ".join(terms)


def transcribe_segments(
    segments: list[dict],
    model: WhisperModel,
    initial_prompt: str | None = None,
) -> list[dict]:
    """Transcribe each {start, end, speaker, text_gold, wav_path} segment,
    cutting audio from its own speaker channel wav. Returns the same dicts
    with an added `text_asr` field (raw ASR hypothesis, pre domain-dict
    correction)."""
    results: list[dict] = []
    for seg in segments:
        audio = load_wav_mono16k(seg["wav_path"])
        start_sample = max(0, int(seg["start"] * SAMPLE_RATE))
        end_sample = min(len(audio), int(seg["end"] * SAMPLE_RATE))
        chunk = audio[start_sample:end_sample] if end_sample > start_sample else np.zeros(1, dtype="float32")

        whisper_segments, _info = model.transcribe(
            chunk,
            language="en",
            initial_prompt=initial_prompt,
            vad_filter=False,  # PriMock57 utterance boundaries are already hand-annotated
        )
        text_asr = "".join(s.text for s in whisper_segments).strip()
        results.append({**seg, "text_asr": text_asr})
    return results


def save_results(results: list[dict], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe PriMock57 utterance segments with faster-whisper (English)")
    parser.add_argument("--consultation", help="Single consultation id, e.g. day1_consultation01")
    parser.add_argument("--all", action="store_true", help="Process every consultation under --processed-root")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--model-size", default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--vocab", default=None, help="domain_dict/vocab_medical.yaml for initial_prompt hint (omit for baseline run)")
    parser.add_argument("--tag", default="raw", help="Output filename tag: asr_<tag>.json (e.g. raw, corrected)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="With --all, only process the first N consultations in sorted order (MVP scope reduction, "
             "see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md: 30 of 57 consultations)",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    if args.all:
        consultation_ids = sorted(p.name for p in processed_root.iterdir() if (p / "segments.json").is_file())
        if args.limit:
            consultation_ids = consultation_ids[: args.limit]
    elif args.consultation:
        consultation_ids = [args.consultation]
    else:
        parser.error("specify --consultation <id> or --all")
        return

    model = load_model(args.model_size)
    initial_prompt = build_initial_prompt(args.vocab)
    if initial_prompt:
        print(f"initial_prompt ({len(initial_prompt.split(', '))} terms): {initial_prompt[:120]}...")

    total_segments = 0
    for consultation_id in consultation_ids:
        seg_path = processed_root / consultation_id / "segments.json"
        with open(seg_path, encoding="utf-8") as f:
            segments = json.load(f)
        results = transcribe_segments(segments, model, initial_prompt=initial_prompt)
        out_path = processed_root / consultation_id / f"asr_{args.tag}.json"
        save_results(results, out_path)
        empty = sum(1 for r in results if not r["text_asr"])
        print(f"{consultation_id}: {len(results)} segments transcribed ({empty} empty) -> {out_path}")
        total_segments += len(results)

    unload_model()
    print(f"OK: {len(consultation_ids)} consultations, {total_segments} segments total")


if __name__ == "__main__":
    main()
