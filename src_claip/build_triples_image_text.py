"""Phase 7, Step 18: build (image, text) triples for the Image-Text CLAIP
branch from IU X-ray (image, report) pairs (parse_iu_xray_reports.py).
Each report is already 1:1 with one image (see that script's docstring), so
this is a thin wrapper around encoding rather than needing any
segment/window alignment logic (unlike build_triples_audio_text.py).

Images are encoded with BiomedCLIP's image tower (ImageTextEncoder,
biomedical-domain-specific), but report text is encoded with the SAME
general-purpose frozen TextEncoder (multilingual-e5-base) used by the
Audio-Text branch -- NOT BiomedCLIP's own text tower. This is what lets
eval_cross_domain_retrieval.py embed one query with a single shared
TextEncoder and route it through either branch's independently trained
text_proj head (see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 4,
Step 10 and Phase 8).

Usage:
    python src_claip/build_triples_image_text.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from encoders_medical import ImageTextEncoder, TextEncoder

DEFAULT_PAIRS_PATH = Path("data/processed/image_text/pairs.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/image_text/claip")


def build_triples(
    pairs: list[dict], image_encoder: ImageTextEncoder, text_encoder: TextEncoder
) -> tuple[list[dict], list, list]:
    triples_meta: list[dict] = []
    image_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []

    for i, pair in enumerate(pairs):
        image_emb = image_encoder.encode_image(pair["image_path"])
        text_emb = text_encoder.encode_text(pair["report_text"])
        triples_meta.append(
            {
                "image_id": pair["image_id"],
                "image_path": pair["image_path"],
                "report_text": pair["report_text"],
                "row": i,
            }
        )
        image_embs.append(image_emb)
        text_embs.append(text_emb)
        if (i + 1) % 200 == 0:
            print(f"  encoded {i + 1}/{len(pairs)} pairs")

    return triples_meta, image_embs, text_embs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image/text embedding triples for the Image-Text CLAIP branch")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    with open(args.pairs, encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"loading encoders (BiomedCLIP image, shared multilingual-e5 text)... {len(pairs)} pairs to encode")
    image_encoder = ImageTextEncoder()
    text_encoder = TextEncoder()

    triples_meta, image_embs, text_embs = build_triples(pairs, image_encoder, text_encoder)

    image_encoder.unload()
    text_encoder.unload()

    image_embs = np.stack(image_embs) if image_embs else np.zeros((0, 1), dtype="float32")
    text_embs = np.stack(text_embs) if text_embs else np.zeros((0, 1), dtype="float32")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "image_emb.npy", image_embs)
    np.save(out_dir / "text_emb.npy", text_embs)
    with open(out_dir / "triples.json", "w", encoding="utf-8") as f:
        json.dump(triples_meta, f, ensure_ascii=False, indent=2)

    print(
        f"OK: {len(triples_meta)} triples -> {out_dir} "
        f"(image_emb {image_embs.shape}, text_emb {text_embs.shape})"
    )


if __name__ == "__main__":
    main()

