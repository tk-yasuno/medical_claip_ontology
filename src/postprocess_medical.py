"""Phase 6, Step 14: English domain-dictionary correction for medical ASR
output. Medical-domain analogue of the bridge CLAIP pipeline's
src/postprocess.py, simplified for English (see
docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Decisions): English text is
already whitespace/punctuation-tokenized, so a SudachiPy morphological
tokenizer (needed in Japanese to avoid substring collisions like
"受けた"->"受桁") is unnecessary here -- a single pass of
word-boundary-anchored regex substitution (`\\b...\\b`) is both simpler and
safe, since English word boundaries don't suffer the same substring-match
problem.

Usage:
    python src/postprocess_medical.py --consultation day1_consultation01 --asr-tag raw
    python src/postprocess_medical.py --all --asr-tag raw --output-tag corrected
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

DEFAULT_VOCAB_PATH = Path("domain_dict/vocab_medical.yaml")
DEFAULT_PROCESSED_ROOT = Path("data/processed/audio_text")


def load_vocab(vocab_path: Path | str = DEFAULT_VOCAB_PATH) -> dict[str, list[str]]:
    vocab_path = Path(vocab_path)
    if not vocab_path.is_file():
        return {}
    with open(vocab_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _build_variant_map(vocab: dict[str, list[str]]) -> dict[str, str]:
    """{lowercased variant: canonical}. Lookup is always done on the
    lowercased matched text (see apply_domain_dict), since the regex match
    itself is case-insensitive and can capture any original casing (e.g.
    a variant appearing capitalized at the start of a sentence)."""
    variant_to_canonical: dict[str, str] = {}
    for canonical, variants in vocab.items():
        for variant in variants or []:
            if variant and variant.lower() != canonical.lower():
                variant_to_canonical[variant.lower()] = canonical
    return variant_to_canonical


def apply_domain_dict(text: str, vocab: dict[str, list[str]]) -> str:
    """Replace domain-dictionary variants in `text` with their canonical
    form, using case-insensitive word-boundary regex matching (see module
    docstring for why this single pass suffices for English, unlike the
    bridge domain's 2-pass SudachiPy approach)."""
    if not text or not vocab:
        return text
    variant_to_canonical = _build_variant_map(vocab)
    if not variant_to_canonical:
        return text

    # Longest-first so multi-word variants are matched before any of their
    # single-word substrings.
    variants_sorted = sorted(variant_to_canonical, key=len, reverse=True)
    pattern = re.compile(
        "|".join(r"\b" + re.escape(v) + r"\b" for v in variants_sorted), re.IGNORECASE
    )
    return pattern.sub(lambda m: variant_to_canonical[m.group(0).lower()], text)


def postprocess_segments(segments: list[dict], vocab: dict[str, list[str]]) -> list[dict]:
    results = []
    for seg in segments:
        corrected = apply_domain_dict(seg.get("text_asr", ""), vocab)
        results.append({**seg, "text_corrected": corrected})
    return results


def save_results(results: list[dict], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply domain_dict/vocab_medical.yaml correction to ASR output")
    parser.add_argument("--consultation", help="Single consultation id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH))
    parser.add_argument("--asr-tag", default="raw", help="Input: asr_<tag>.json")
    parser.add_argument("--output-tag", default="corrected", help="Output: asr_<tag>.json (adds text_corrected field)")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    if args.all:
        consultation_ids = sorted(p.name for p in processed_root.iterdir() if (p / f"asr_{args.asr_tag}.json").is_file())
    elif args.consultation:
        consultation_ids = [args.consultation]
    else:
        parser.error("specify --consultation <id> or --all")
        return

    vocab = load_vocab(args.vocab)
    print(f"loaded {sum(len(v) for v in vocab.values())} variant mappings for {len(vocab)} canonical terms from {args.vocab}")

    total_corrected = 0
    total_segments = 0
    for consultation_id in consultation_ids:
        in_path = processed_root / consultation_id / f"asr_{args.asr_tag}.json"
        with open(in_path, encoding="utf-8") as f:
            segments = json.load(f)
        results = postprocess_segments(segments, vocab)
        out_path = processed_root / consultation_id / f"asr_{args.output_tag}.json"
        save_results(results, out_path)
        corrected = sum(1 for r in results if r["text_corrected"] != r["text_asr"])
        total_corrected += corrected
        total_segments += len(results)

    print(f"OK: {total_segments} segments, {total_corrected} corrected by domain dict -> asr_{args.output_tag}.json")


if __name__ == "__main__":
    main()
