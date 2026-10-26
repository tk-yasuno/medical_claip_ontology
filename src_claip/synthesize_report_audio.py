"""v0.2.0, Phase A: synthesize spoken-narration audio for IU X-ray reports
using Piper TTS (neural, offline, ONNX-based -- runs under its own
`.venv-tts` to avoid any dependency conflict with `.venv-claip`'s pinned
torch/transformers stack, same "isolate conflicting dependencies into
their own venv" philosophy as the bridge domain's `.venv-asr`/`.venv-claip`
split).

Rationale (see docs/Plan_v0.2.0_Medical_Ontology_CLAIP.md): no instantly
downloadable public dataset pairs real human speech with chest-X-ray
images in the same domain (the one dataset that does -- PhysioNet's
"Eye Gaze Data for Chest Xrays", built on MIMIC-CXR -- requires
PhysioNet's Credentialed Health Data License: CITI training + a signed
DUA + a multi-day approval process, which fails this MVP's
"immediately downloadable" requirement). Synthesizing narration of the
SAME IU X-ray report text that is already paired with its image
guarantees image/audio/text alignment *and* single-domain vocabulary by
construction, enabling a genuine 3-modality (image, audio, text) CLAIP
triple -- unlike v0.1.x's two disjoint 2-modality branches (PriMock57
audio-text vs. IU X-ray image-text), which had almost no shared
vocabulary (see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md v0.1.1 addendum).

Usage (run under .venv-tts):
    .venv-tts\\Scripts\\python.exe src_claip\\synthesize_report_audio.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from piper.voice import PiperVoice

DEFAULT_PAIRS_PATH = Path("data/processed/image_text/pairs.json")
DEFAULT_AUDIO_DIR = Path("data/processed/image_text/audio")
DEFAULT_VOICE_MODEL = Path("data/tts_voices/en_US-lessac-medium.onnx")
DEFAULT_VOICE_CONFIG = Path("data/tts_voices/en_US-lessac-medium.onnx.json")
TARGET_SAMPLE_RATE = 16000  # matches encoders_medical.AudioEncoder's SAMPLE_RATE


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear-interpolation resampler (no extra scipy/soundfile
    dependency needed in the lightweight .venv-tts): good enough for
    feeding a frozen Whisper encoder, not for perceptual audio quality."""
    if src_rate == dst_rate:
        return audio
    duration = len(audio) / src_rate
    n_dst = int(round(duration * dst_rate))
    src_times = np.linspace(0, duration, num=len(audio), endpoint=False)
    dst_times = np.linspace(0, duration, num=n_dst, endpoint=False)
    return np.interp(dst_times, src_times, audio.astype(np.float32)).astype(np.int16)


def synthesize_pair_audio(
    voice: PiperVoice, text: str, out_path: Path, target_rate: int = TARGET_SAMPLE_RATE
) -> None:
    import wave

    chunks = list(voice.synthesize(text))
    if not chunks:
        audio = np.zeros(1, dtype=np.int16)
        src_rate = target_rate
    else:
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        src_rate = chunks[0].sample_rate

    audio = resample_linear(audio, src_rate, target_rate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_rate)
        wf.writeframes(audio.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize narration audio for IU X-ray reports (Piper TTS)")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS_PATH))
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--voice-model", default=str(DEFAULT_VOICE_MODEL))
    parser.add_argument("--voice-config", default=str(DEFAULT_VOICE_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.pairs, encoding="utf-8") as f:
        pairs = json.load(f)

    voice = PiperVoice.load(args.voice_model, args.voice_config)
    audio_dir = Path(args.audio_dir)

    for i, pair in enumerate(pairs):
        out_path = audio_dir / f"{pair['image_id']}.wav"
        if out_path.is_file() and not args.overwrite:
            continue
        synthesize_pair_audio(voice, pair["report_text"], out_path)
        if (i + 1) % 100 == 0:
            print(f"  synthesized {i + 1}/{len(pairs)}")

    print(f"OK: synthesized narration audio for {len(pairs)} IU X-ray reports -> {audio_dir}")


if __name__ == "__main__":
    main()
