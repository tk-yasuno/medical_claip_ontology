"""Phase 3/6, Step 8 & 16: Word Error Rate evaluation against PriMock57's
human-transcribed `text_gold`, the ground-truth comparison the bridge
CLAIP pipeline never had (see
docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Decisions/TL;DR).

Computes both:
  - corpus-level micro-averaged WER (all segments pooled, weighted by
    reference length, which is what jiwer.wer() does natively)
  - a "medical-term subset" WER restricted to segments whose text_gold
    mentions at least one term from domain_dict/medical_ontology.yaml,
    to check whether domain-dictionary correction specifically helps on
    medically-relevant utterances (not just overall transcript quality)

Usage:
    python src/wer_eval_medical.py --asr-tag raw --output wer_raw.json
    python src/wer_eval_medical.py --asr-tag corrected --text-field text_corrected --output wer_corrected.json
    python src/wer_eval_medical.py --compare data/processed/audio_text/_reports/wer_raw.json data/processed/audio_text/_reports/wer_corrected.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import yaml

DEFAULT_PROCESSED_ROOT = Path("data/processed/audio_text")
DEFAULT_ONTOLOGY_PATH = Path("domain_dict/medical_ontology.yaml")
DEFAULT_REPORT_DIR = Path("output")


def load_ontology_terms(ontology_path: Path | str = DEFAULT_ONTOLOGY_PATH) -> set[str]:
    ontology_path = Path(ontology_path)
    if not ontology_path.is_file():
        return set()
    with open(ontology_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    terms: set[str] = set()
    for category_terms in data.values():
        terms.update(t.lower() for t in (category_terms or []))
    return terms


def mentions_medical_term(text: str, ontology_terms: set[str]) -> bool:
    text_lower = text.lower()
    return any(term in text_lower for term in ontology_terms)


def evaluate(
    processed_root: Path,
    consultation_ids: list[str],
    asr_tag: str,
    text_field: str,
    ontology_terms: set[str],
) -> dict:
    references: list[str] = []
    hypotheses: list[str] = []
    medical_references: list[str] = []
    medical_hypotheses: list[str] = []

    for consultation_id in consultation_ids:
        asr_path = processed_root / consultation_id / f"asr_{asr_tag}.json"
        if not asr_path.is_file():
            continue
        with open(asr_path, encoding="utf-8") as f:
            segments = json.load(f)
        for seg in segments:
            ref = (seg.get("text_gold") or "").strip()
            hyp = (seg.get(text_field) or "").strip()
            if not ref:
                continue  # jiwer requires a non-empty reference per segment
            references.append(ref)
            hypotheses.append(hyp or " ")
            if mentions_medical_term(ref, ontology_terms):
                medical_references.append(ref)
                medical_hypotheses.append(hyp or " ")

    overall_wer = jiwer.wer(references, hypotheses) if references else None
    medical_wer = jiwer.wer(medical_references, medical_hypotheses) if medical_references else None

    return {
        "asr_tag": asr_tag,
        "text_field": text_field,
        "n_segments": len(references),
        "overall_wer": overall_wer,
        "n_medical_segments": len(medical_references),
        "medical_subset_wer": medical_wer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WER of ASR output against PriMock57 manual transcripts")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    parser.add_argument("--asr-tag", default="raw")
    parser.add_argument("--text-field", default="text_asr", help="text_asr (pre-correction) or text_corrected (post domain-dict)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: output/wer_<asr-tag>.json)")
    parser.add_argument("--compare", nargs=2, metavar=("WER_A_JSON", "WER_B_JSON"), help="Print a before/after comparison of two wer_eval_medical.py outputs")
    args = parser.parse_args()

    if args.compare:
        with open(args.compare[0], encoding="utf-8") as f:
            report_a = json.load(f)
        with open(args.compare[1], encoding="utf-8") as f:
            report_b = json.load(f)
        print(f"{'':20s} {'overall_wer':>12s} {'medical_subset_wer':>20s} {'n_segments':>12s}")
        for tag, report in (("A (" + args.compare[0] + ")", report_a), ("B (" + args.compare[1] + ")", report_b)):
            print(f"{tag:20s} {report['overall_wer']:>12.4f} {report['medical_subset_wer']:>20.4f} {report['n_segments']:>12d}")
        delta_overall = report_b["overall_wer"] - report_a["overall_wer"]
        delta_medical = report_b["medical_subset_wer"] - report_a["medical_subset_wer"]
        print(f"delta (B-A): overall_wer={delta_overall:+.4f}, medical_subset_wer={delta_medical:+.4f}")
        return

    processed_root = Path(args.processed_root)
    consultation_ids = sorted(p.name for p in processed_root.iterdir() if (p / f"asr_{args.asr_tag}.json").is_file())
    ontology_terms = load_ontology_terms(args.ontology)

    report = evaluate(processed_root, consultation_ids, args.asr_tag, args.text_field, ontology_terms)

    output_path = Path(args.output) if args.output else DEFAULT_REPORT_DIR / f"wer_{args.asr_tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"OK: {report['n_segments']} segments, overall WER={report['overall_wer']:.4f}, "
        f"medical-term subset ({report['n_medical_segments']} segments) WER={report['medical_subset_wer']} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
