"""Phase 4, Step 9: build the (image, report-text) pair list for the
Image-Text CLAIP branch from the Hugging Face mirror of the IU X-ray /
Open-i chest X-ray collection (MLforHealthcare/Indiana_University_Chest_
X-ray_Collection -- MIT-licensed mirror, no login required).

Note (see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Decisions): the
official NLM openi.nlm.nih.gov direct-download tgz links referenced in the
plan no longer serve the dataset (the NLM site has since migrated to an
Angular SPA at openi-vip.nlm.nih.gov and the old collection URLs now return
an HTML error page instead of the tarball). This script therefore sources
the same underlying dataset (7,470 images / 3,955 reports, CC BY-NC-ND
provenance) from the Hugging Face parquet mirror instead of the raw NLM
XML+PNG archives; each parquet row is already one (image, report) pair
(`image`: {bytes, path}, `question`, `report` columns), so no XML/
parentImage parsing is needed -- a simplification over the originally
planned XML route, not a change in dataset content.

Usage:
    python src_claip/parse_iu_xray_reports.py --n-pairs 1000
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

DEFAULT_PARQUET_PATHS = [
    "data/iu_xray/data/test-00000-of-00001.parquet",
    "data/iu_xray/data/train-00000-of-00003.parquet",
]
DEFAULT_IMAGES_DIR = Path("data/iu_xray/images")
DEFAULT_OUTPUT_PATH = Path("data/processed/image_text/pairs.json")
DEFAULT_SEED = 42


def load_pairs_table(parquet_paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in parquet_paths]
    df = pd.concat(frames, ignore_index=True)
    # The report column already combines "FINDINGS: ... IMPRESSION: ..."
    # (see the HF dataset card); drop rows where both sections are empty.
    df = df[df["report"].str.strip().str.len() > len("FINDINGS: . IMPRESSION:")]
    return df.reset_index(drop=True)


def export_pairs(
    df: pd.DataFrame,
    n_pairs: int,
    images_dir: Path | str = DEFAULT_IMAGES_DIR,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Subsample `n_pairs` rows (fixed seed for reproducibility), decode
    each row's embedded PNG bytes to a file under `images_dir`, and return
    the pair list `{"image_id", "image_path", "report_text"}`."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    sampled = df.sample(n=min(n_pairs, len(df)), random_state=seed).reset_index(drop=True)

    pairs: list[dict] = []
    for i, row in sampled.iterrows():
        image_id = f"iuxray_{i:05d}"
        image_path = images_dir / f"{image_id}.png"
        if not image_path.is_file():
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            img.save(image_path)
        pairs.append({"image_id": image_id, "image_path": str(image_path), "report_text": row["report"]})
    return pairs


def save_pairs(pairs: list[dict], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IU X-ray (image, report) pairs for the Image-Text CLAIP branch")
    parser.add_argument("--parquet", nargs="+", default=DEFAULT_PARQUET_PATHS)
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    df = load_pairs_table(args.parquet)
    print(f"loaded {len(df)} usable (image, report) rows from {len(args.parquet)} parquet shard(s)")

    pairs = export_pairs(df, args.n_pairs, images_dir=args.images_dir, seed=args.seed)
    save_pairs(pairs, args.output)
    print(f"OK: {len(pairs)} pairs exported -> {args.output} (images under {args.images_dir})")


if __name__ == "__main__":
    main()
