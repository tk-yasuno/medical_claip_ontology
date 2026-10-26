"""Phase 5, Step 13: discover genuine faster-whisper ASR misrecognitions of
medical-ontology terms by diffing raw ASR output against PriMock57's
human-transcribed `text_gold`, then seed domain_dict/vocab_medical.yaml
with the real variants found (medical-domain analogue of the bridge CLAIP
pipeline's empirical vocab-curation approach documented in
domain_dict/vocab.yaml's header comment -- "found a real homophone
collision during pilot testing", not a guessed list).

Only substitutions where the *reference* (gold) word/phrase matches a
domain_dict/medical_ontology.yaml term are kept as candidates: this is
what distinguishes "faster-whisper mis-transcribed a medical term" (worth
correcting) from "ASR made some unrelated error" (out of scope here).

Usage:
    python src/vocab_merge_medical.py --asr-tag raw
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

import yaml

DEFAULT_PROCESSED_ROOT = Path("data/processed/audio_text")
DEFAULT_ONTOLOGY_PATH = Path("domain_dict/medical_ontology.yaml")
DEFAULT_VOCAB_PATH = Path("domain_dict/vocab_medical.yaml")

MIN_VARIANT_FREQ = 1
_WORD_PATTERN = re.compile(r"[A-Za-z']+")

# 【実データ検証で判明した方針、橋梁版vocab.yamlの「同音異義語の一般語」除外方針と同じ考え方】
# diff-based discovery can surface two kinds of false-positive risk:
#   (1) the "variant" is itself a generic, independently-common English word
#       (e.g. "right", "arm", "moment", "person"): blindly replacing every
#       future occurrence of that word with the canonical medical term would
#       corrupt unrelated, correctly-transcribed speech far more often than
#       it fixes a real misrecognition.
#   (2) the "variant" is itself ANOTHER valid ontology term under a
#       different canonical (e.g. gold "asthma" aligned against ASR "acne" --
#       "acne" is a real, distinct skin condition, not a misrecognition of
#       "asthma"; this is diff/alignment noise, not a genuine homophone).
# Both are excluded below rather than merged blindly (see merge_into_vocab()).
RISKY_GENERIC_VARIANTS = {
    "right", "arm", "moment", "person", "every person", "rumbling",
    # Not yet in this MVP's small 122-term ontology list, but a real,
    # distinct, well-known medical condition name in its own right (not a
    # phonetic variant of "asthma") -- blindly auto-correcting it would be
    # actively harmful, so it is excluded by name regardless of ontology
    # membership.
    "acne",
}


def load_ontology_terms(ontology_path: Path | str = DEFAULT_ONTOLOGY_PATH) -> set[str]:
    ontology_path = Path(ontology_path)
    with open(ontology_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    terms: set[str] = set()
    for category_terms in data.values():
        terms.update(t.lower() for t in (category_terms or []) if len(t.split()) <= 3)
    return terms


def _tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def find_term_misrecognitions(gold: str, asr: str, ontology_terms: set[str]) -> list[tuple[str, str]]:
    """Word-level diff `gold` (reference) against `asr` (hypothesis); for
    every 'replace'/'delete' opcode block where the reference phrase
    matches a known ontology term, return (canonical_term, asr_variant)
    pairs."""
    gold_words = _tokenize(gold)
    asr_words = _tokenize(asr)
    if not gold_words or not asr_words:
        return []

    matcher = difflib.SequenceMatcher(None, gold_words, asr_words)
    found: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("replace", "delete"):
            continue
        gold_phrase = " ".join(gold_words[i1:i2])
        if gold_phrase not in ontology_terms:
            continue
        asr_phrase = " ".join(asr_words[j1:j2]).strip()
        if not asr_phrase or asr_phrase == gold_phrase:
            continue
        found.append((gold_phrase, asr_phrase))
    return found


def discover_variants(
    processed_root: Path, consultation_ids: list[str], asr_tag: str, ontology_terms: set[str]
) -> dict[str, Counter]:
    """{canonical_term: Counter({asr_variant: freq})}."""
    discoveries: dict[str, Counter] = {}
    for consultation_id in consultation_ids:
        asr_path = processed_root / consultation_id / f"asr_{asr_tag}.json"
        if not asr_path.is_file():
            continue
        with open(asr_path, encoding="utf-8") as f:
            segments = json.load(f)
        for seg in segments:
            gold, asr = seg.get("text_gold", ""), seg.get("text_asr", "")
            for canonical, variant in find_term_misrecognitions(gold, asr, ontology_terms):
                discoveries.setdefault(canonical, Counter())[variant] += 1
    return discoveries


def merge_into_vocab(
    discoveries: dict[str, Counter],
    vocab_path: Path | str = DEFAULT_VOCAB_PATH,
    min_freq: int = MIN_VARIANT_FREQ,
    ontology_terms: set[str] | None = None,
) -> dict[str, list[str]]:
    vocab_path = Path(vocab_path)
    vocab: dict[str, list[str]] = {}
    if vocab_path.is_file():
        with open(vocab_path, encoding="utf-8") as f:
            vocab = yaml.safe_load(f) or {}
    ontology_terms = ontology_terms or set()

    merged = 0
    rejected_risky = 0
    for canonical, variant_counts in discoveries.items():
        existing = set(v.lower() for v in vocab.get(canonical, []))
        for variant, freq in variant_counts.items():
            if freq < min_freq or variant.lower() in existing or variant.lower() == canonical.lower():
                continue
            # Safety filters (see RISKY_GENERIC_VARIANTS docstring above):
            # skip generic independently-common words, and skip any variant
            # that is itself a distinct, valid ontology term (diff/alignment
            # noise, not a genuine homophone misrecognition).
            if variant.lower() in RISKY_GENERIC_VARIANTS or variant.lower() in ontology_terms:
                rejected_risky += 1
                continue
            vocab.setdefault(canonical, []).append(variant)
            merged += 1
    if rejected_risky:
        print(f"rejected {rejected_risky} risky/generic-word or cross-ontology-collision variants (not merged)")

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# domain_dict/vocab_medical.yaml -- ASR misrecognition correction dictionary (English)\n"
        "#\n"
        "# Format: canonical_term: [observed faster-whisper misrecognition variants]\n"
        "# Variants here are NOT guessed: each was found by diffing faster-whisper's raw ASR\n"
        "# output against PriMock57's human-transcribed text_gold, restricted to cases where the\n"
        "# reference word/phrase is a domain_dict/medical_ontology.yaml term (see\n"
        "# src/vocab_merge_medical.py and docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 5).\n"
        "# postprocess_medical.py replaces variants with their canonical form via case-insensitive\n"
        "# word-boundary regex (no morphological tokenizer needed for English; see Decisions).\n\n"
    )
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(vocab, f, allow_unicode=True, sort_keys=True)

    print(f"merged {merged} new (canonical, variant) pairs into {vocab_path}")
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover real ASR misrecognitions of medical ontology terms")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH))
    parser.add_argument("--asr-tag", default="raw")
    parser.add_argument("--min-freq", type=int, default=MIN_VARIANT_FREQ)
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    consultation_ids = sorted(p.name for p in processed_root.iterdir() if (p / f"asr_{args.asr_tag}.json").is_file())
    ontology_terms = load_ontology_terms(args.ontology)
    print(f"scanning {len(consultation_ids)} consultations for misrecognitions of {len(ontology_terms)} ontology terms...")

    discoveries = discover_variants(processed_root, consultation_ids, args.asr_tag, ontology_terms)
    n_variants = sum(len(c) for c in discoveries.values())
    print(f"found {len(discoveries)} terms with {n_variants} distinct misrecognition variants")
    for canonical, counter in sorted(discoveries.items()):
        for variant, freq in counter.most_common():
            print(f"  {canonical!r} -> {variant!r} (x{freq})")

    merge_into_vocab(discoveries, args.vocab, min_freq=args.min_freq, ontology_terms=ontology_terms)


if __name__ == "__main__":
    main()
