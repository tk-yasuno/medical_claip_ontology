"""Medical-domain CLAIP encoders (docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md
Phase 4, Step 10): same frozen-encoder-plus-common-interface design as the
bridge-domain src_claip/encoders.py, with only the image<->text encoder
swapped for a biomedical-domain model.

- ImageTextEncoder: BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-
  vit_base_patch16_224, MIT license, no login required), loaded via
  open_clip. Replaces the bridge domain's rinna japanese-clip-vit-b-16 --
  same `encode_image()`/`encode_text()` interface, so
  build_triples_image_text.py doesn't need to know which CLIP variant is
  loaded underneath.
- AudioEncoder, TextEncoder: **unchanged, copied verbatim** from
  src_claip/encoders.py. transformers.WhisperModel (encoder-only) and
  sentence-transformers multilingual-e5-base were never Japanese-specific
  to begin with, so reusing them for English PriMock57 audio/text costs
  zero code changes -- itself part of this MVP's "CLAIP generalizes across
  domain AND language" argument (see Plan Decisions).

Usage: imported, not run directly.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

IMAGE_TEXT_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
AUDIO_MODEL_NAME = "openai/whisper-small"
TEXT_MODEL_NAME = "intfloat/multilingual-e5-base"
SAMPLE_RATE = 16000
WHISPER_WINDOW_SEC = 30.0
BIOMEDCLIP_CONTEXT_LENGTH = 256  # matches the "_256_" in the model name


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class ImageTextEncoder:
    """Wraps BiomedCLIP (open_clip) for both chest X-ray image and English
    medical-report text embeddings, sharing BiomedCLIP's own pretrained
    joint space (PMC-15M image-text contrastive pretraining)."""

    def __init__(self, device: str | None = None):
        import open_clip

        self.device = device or _default_device()
        self.model, _preprocess_train, self.preprocess = open_clip.create_model_and_transforms(
            IMAGE_TEXT_MODEL_NAME
        )
        self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(IMAGE_TEXT_MODEL_NAME)

    def encode_image(self, image_path: Path | str) -> np.ndarray:
        from PIL import Image

        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
        return features.squeeze(0).cpu().numpy().astype("float32")

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text or " "], context_length=BIOMEDCLIP_CONTEXT_LENGTH).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features.squeeze(0).cpu().numpy().astype("float32")

    def unload(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AudioEncoder:
    """Wraps the encoder half of a (small, frozen) transformers Whisper
    model -- unchanged from src_claip/encoders.py. See that file's
    docstring for the mean-pooling / 30s-window rationale."""

    def __init__(self, device: str | None = None):
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.device = device or _default_device()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
        self.model = WhisperModel.from_pretrained(AUDIO_MODEL_NAME).to(self.device)
        self.model.eval()
        self.encoder = self.model.get_encoder()

    def encode_audio_array(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            hidden_states = self.encoder(inputs).last_hidden_state  # (1, T, hidden_dim)
        return hidden_states.mean(dim=1).squeeze(0).cpu().numpy().astype("float32")

    def encode_audio_segment(self, wav_path: Path | str, start: float, end: float) -> np.ndarray:
        audio = _load_wav_mono16k(wav_path)
        start_sample = max(0, int(start * SAMPLE_RATE))
        end_sample = min(len(audio), int(end * SAMPLE_RATE))
        max_samples = int(WHISPER_WINDOW_SEC * SAMPLE_RATE)
        chunk = audio[start_sample : min(end_sample, start_sample + max_samples)]
        if len(chunk) == 0:
            chunk = np.zeros(1, dtype="float32")
        return self.encode_audio_array(chunk)

    def encode_audio_file(self, wav_path: Path | str) -> np.ndarray:
        """Encode an entire wav file (e.g. a TTS-synthesized narration with
        no natural utterance boundaries, see
        src_claip/synthesize_report_audio.py), truncated to the first
        WHISPER_WINDOW_SEC (30s) like encode_audio_segment()."""
        audio = _load_wav_mono16k(wav_path)
        max_samples = int(WHISPER_WINDOW_SEC * SAMPLE_RATE)
        chunk = audio[:max_samples] if len(audio) > 0 else np.zeros(1, dtype="float32")
        return self.encode_audio_array(chunk)

    def unload(self) -> None:
        del self.model, self.encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TextEncoder:
    """Wraps sentence-transformers multilingual-e5-base -- unchanged from
    src_claip/encoders.py. See that file's docstring for the "query: "
    prefix rationale."""

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.device = device or _default_device()
        self.model = SentenceTransformer(TEXT_MODEL_NAME, device=self.device)

    def encode_text(self, text: str) -> np.ndarray:
        embedding = self.model.encode(f"query: {text or ' '}", normalize_embeddings=False)
        return np.asarray(embedding, dtype="float32")

    def unload(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_wav_cache: dict[str, np.ndarray] = {}


def _load_wav_mono16k(path: Path | str) -> np.ndarray:
    path = str(path)
    if path in _wav_cache:
        return _wav_cache[path]
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
            raise ValueError(f"{path} is not 16kHz mono (got {wf.getframerate()}Hz, {wf.getnchannels()}ch)")
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
    _wav_cache[path] = audio
    return audio
