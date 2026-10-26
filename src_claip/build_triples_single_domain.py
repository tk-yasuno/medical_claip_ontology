"""v0.2.0: build true (image, audio, text) triples for a single-domain
(chest radiology) CLAIP model -- the architectural fix for v0.1.x's two
disjoint 2-modality branches (PriMock57 Audio-Text vs. IU X-ray
Image-Text), which had almost no shared vocabulary (see
docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md v0.1.1 addendum). Each of the
1,000 IU X-ray (image, report) pairs now also has a TTS-synthesized
narration audio of its own report_text (synthesize_report_audio.py), so
all 3 modalities describe the exact same case -- a genuine same-domain,
same-case triple, matching the bridge domain's original 3-modality design.

Usage:
    python src_claip/build_triples_single_domain.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from encoders_medical import AudioEncoder, ImageTextEncoder, TextEncoder

DEFAULT_PAIRS_PATH = Path("data/processed/image_text/pairs.json")
DEFAULT_AUDIO_DIR = Path("data/processed/image_text/audio")
DEFAULT_OUTPUT_DIR = Path("data/processed/image_text/claip_v0.2.0")


def build_triples(
    pairs: list[dict],
    audio_dir: Path,
    image_encoder: ImageTextEncoder,
    audio_encoder: AudioEncoder,
    text_encoder: TextEncoder,
) -> tuple[list[dict], list, list, list]:
    triples_meta: list[dict] = []
    image_embs: list[np.ndarray] = []
    audio_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []
    skipped = 0

    for i, pair in enumerate(pairs):
        audio_path = audio_dir / f"{pair['image_id']}.wav"
        if not audio_path.is_file():
            skipped += 1
            continue

        image_emb = image_encoder.encode_image(pair["image_path"])
        audio_emb = audio_encoder.encode_audio_file(audio_path)
        text_emb = text_encoder.encode_text(pair["report_text"])

        triples_meta.append(
            {
                "image_id": pair["image_id"],
                "image_path": pair["image_path"],
                "audio_path": str(audio_path),
                "report_text": pair["report_text"],
                "row": len(triples_meta),
            }
        )
        image_embs.append(image_emb)
        audio_embs.append(audio_emb)
        text_embs.append(text_emb)
        if (i + 1) % 200 == 0:
            print(f"  encoded {i + 1}/{len(pairs)} pairs")

    if skipped:
        print(f"  skipped {skipped} pairs with no synthesized audio (run synthesize_report_audio.py first)")
    return triples_meta, image_embs, audio_embs, text_embs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build (image, audio, text) triples for the single-domain IU X-ray CLAIP model")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS_PATH))
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    with open(args.pairs, encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"loading encoders (BiomedCLIP image, Whisper audio, multilingual-e5 text)... {len(pairs)} pairs to encode")
    image_encoder = ImageTextEncoder()
    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()

    triples_meta, image_embs, audio_embs, text_embs = build_triples(
        pairs, Path(args.audio_dir), image_encoder, audio_encoder, text_encoder
    )

    image_encoder.unload()
    audio_encoder.unload()
    text_encoder.unload()

    image_embs = np.stack(image_embs) if image_embs else np.zeros((0, 1), dtype="float32")
    audio_embs = np.stack(audio_embs) if audio_embs else np.zeros((0, 1), dtype="float32")
    text_embs = np.stack(text_embs) if text_embs else np.zeros((0, 1), dtype="float32")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "image_emb.npy", image_embs)
    np.save(out_dir / "audio_emb.npy", audio_embs)
    np.save(out_dir / "text_emb.npy", text_embs)
    with open(out_dir / "triples.json", "w", encoding="utf-8") as f:
        json.dump(triples_meta, f, ensure_ascii=False, indent=2)

    print(
        f"OK: {len(triples_meta)} triples -> {out_dir} "
        f"(image_emb {image_embs.shape}, audio_emb {audio_embs.shape}, text_emb {text_embs.shape})"
    )


if __name__ == "__main__":
    main()
