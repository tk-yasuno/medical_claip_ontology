"""Phase 5, Step 12: build the Medical Ontology (domain_dict/medical_ontology.yaml,
Disease / Symptom / Procedure / Drug / Body-Part) from PriMock57's clinician-written
consultation notes and IU X-ray's radiology reports.

Candidate sources (see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 5,
Decisions -- deliberately NOT "ask the LLM to read free text and propose a
diagnosis"):
  - Audio side: `highlights` + `presenting_complaint` from every
    PriMock57 `notes/*.json` -- phrases the consulting clinician
    *themselves* flagged as important medical information. Already
    vetted by construction, unlike noun phrases mined from a raw
    transcript (contrast with the bridge domain's term_extraction.py,
    which has no such pre-vetted source and must mine candidates itself).
  - Image side: a fixed list of common chest X-ray finding/anatomy terms
    (FORCE_INCLUDE_XRAY_TERMS below), checked for presence in each IU
    X-ray report_text. (The bridge domain's ontology_vocab_import.py had
    a pre-verified 6,745-triple corpus to bulk-import from; no equivalent
    structured, license-clear resource exists for this MVP's IU X-ray
    mirror, so a hand-curated candidate list is the pragmatic substitute
    -- see Plan Decisions.)

A local Ollama LLM (qwen2.5:7b-instruct-q4_k_m, same model as the bridge
domain) is used ONLY to classify each candidate phrase into one of the 5
ontology categories (or reject it as not domain vocabulary) -- it is never
asked to infer a diagnosis from free text.

Usage:
    python src/term_extraction_medical.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import requests
import yaml

DEFAULT_NOTES_DIR = Path("data/primock57/notes")
DEFAULT_IUXRAY_PAIRS = Path("data/processed/image_text/pairs.json")
DEFAULT_OUTPUT_PATH = Path("domain_dict/medical_ontology.yaml")
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_k_m"
DEFAULT_HOST = "http://localhost:11434"

MIN_TERM_LENGTH = 3
MAX_TERM_LENGTH = 60
CATEGORIES = ["Disease", "Symptom", "Procedure", "Drug", "Body-Part"]
_CONFIDENCE_LEVELS = {"high", "medium", "low"}

# Common chest X-ray finding/anatomy vocabulary, checked for literal
# presence (case-insensitive) in each IU X-ray report_text. Hand-curated
# because no bulk-verified ontology source (equivalent to the bridge
# domain's damage_cause_ontology) exists for this dataset (see Decisions).
FORCE_INCLUDE_XRAY_TERMS = [
    "pneumonia", "cardiomegaly", "atelectasis", "pleural effusion", "effusion",
    "consolidation", "opacity", "nodule", "emphysema", "pulmonary edema",
    "edema", "pneumothorax", "fracture", "calcification", "granuloma",
    "hernia", "scoliosis", "hyperinflation", "fibrosis", "COPD",
    "cardiopulmonary", "lung", "lungs", "heart", "mediastinum", "diaphragm",
    "pleura", "rib", "ribs", "spine", "aorta", "hilar",
]

COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "no", "none", "normal", "within normal limits",
}

# Common primary-care/radiology procedure vocabulary, checked for literal
# presence (case-insensitive, word-boundary) in PriMock57 notes' full
# `note` body (not just `highlights`, which rarely flags procedures) and
# IU X-ray report_text (the imaging procedure itself). Same
# "keyword-presence check, not LLM free-text diagnosis" approach as
# FORCE_INCLUDE_XRAY_TERMS above (see Decisions): this is what seeds the
# Procedure category, which audio-side highlights alone rarely populate.
FORCE_INCLUDE_PROCEDURE_TERMS = [
    "chest x-ray", "x-ray", "radiograph", "ECG", "blood test", "urine test",
    "urinalysis", "referral", "A&E", "inhaler", "conservative management",
    "follow-up", "antibiotics", "vaccination", "physiotherapy", "surgery",
    "biopsy", "CT scan", "MRI", "ultrasound", "catheter", "nebuliser",
    "oxygen therapy", "blood pressure check", "swab",
]

TERM_CLASSIFY_PROMPT_TEMPLATE = """You are a clinical terminology expert. Below is a list of candidate terms
extracted from primary-care consultation notes and chest X-ray radiology reports. For each term, output exactly
one line in the format "term|is_term|category|confidence" (pipe-separated, no other text):

(1) term: copied exactly as given
(2) is_term: "yes" if the term is itself a genuine Disease, Symptom, Procedure, Drug, or Body-Part/anatomical
    structure name (not a generic word like "normal", "patient", "today"); otherwise "no"
(3) category: exactly one of Disease, Symptom, Procedure, Drug, Body-Part (leave empty if is_term is "no")
(4) confidence: "high" only if you are confident the term is correctly categorized; otherwise "medium" or "low"

Do not diagnose or interpret any patient case. Only classify the terms themselves as vocabulary.

--- candidate terms ---
{terms}
--- end ---"""


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, host: str | None = None, timeout: int = 300) -> str:
    host = host or DEFAULT_HOST
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    response = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"].strip()


def collect_audio_side_candidates(notes_dir: Path | str = DEFAULT_NOTES_DIR) -> Counter:
    """Frequency-count `highlights` + `presenting_complaint` phrases across
    every PriMock57 note (see module docstring: already clinician-vetted,
    no NLP phrase mining needed)."""
    notes_dir = Path(notes_dir)
    counter: Counter[str] = Counter()
    for note_path in sorted(notes_dir.glob("*.json")):
        with open(note_path, encoding="utf-8") as f:
            note = json.load(f)
        candidates = list(note.get("highlights") or [])
        pc = (note.get("presenting_complaint") or "").strip()
        if pc:
            candidates.append(pc)
        for phrase in candidates:
            phrase = re.sub(r"\s+", " ", phrase).strip().strip(".")
            if MIN_TERM_LENGTH <= len(phrase) <= MAX_TERM_LENGTH and phrase.lower() not in COMMON_STOPWORDS:
                counter[phrase] += 1
    return counter


def collect_procedure_candidates(
    notes_dir: Path | str = DEFAULT_NOTES_DIR, pairs_path: Path | str = DEFAULT_IUXRAY_PAIRS
) -> Counter:
    """Count literal (case-insensitive) occurrences of
    FORCE_INCLUDE_PROCEDURE_TERMS across PriMock57 notes' full `note` body
    and IU X-ray report_text (see module docstring)."""
    notes_dir = Path(notes_dir)
    pairs_path = Path(pairs_path)
    texts: list[str] = []
    for note_path in sorted(notes_dir.glob("*.json")):
        with open(note_path, encoding="utf-8") as f:
            note = json.load(f)
        texts.append(note.get("note") or "")
    if pairs_path.is_file():
        with open(pairs_path, encoding="utf-8") as f:
            pairs = json.load(f)
        texts.extend(pair.get("report_text", "") for pair in pairs)

    counter: Counter[str] = Counter()
    for term in FORCE_INCLUDE_PROCEDURE_TERMS:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        freq = sum(1 for text in texts if pattern.search(text))
        if freq > 0:
            counter[term] = freq
    return counter


def collect_image_side_candidates(pairs_path: Path | str = DEFAULT_IUXRAY_PAIRS) -> Counter:
    """Count literal (case-insensitive) occurrences of
    FORCE_INCLUDE_XRAY_TERMS across every IU X-ray report_text."""
    pairs_path = Path(pairs_path)
    counter: Counter[str] = Counter()
    if not pairs_path.is_file():
        return counter
    with open(pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)
    for term in FORCE_INCLUDE_XRAY_TERMS:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        freq = sum(1 for pair in pairs if pattern.search(pair.get("report_text", "")))
        if freq > 0:
            counter[term] = freq
    return counter


def _parse_classify_response(raw: str, expected_terms: set[str]) -> dict[str, tuple[bool, str, str]]:
    """Parse "term|is_term|category|confidence" lines, matching the
    bridge domain's term_extraction.py robustness approach (scan from the
    right for a recognized confidence token, since local 7B models
    sometimes shift fields)."""
    results: dict[str, tuple[bool, str, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        term, is_term_text = parts[0], parts[1]
        if term not in expected_terms:
            continue
        confidence = "low"
        for p in reversed(parts[2:]):
            if p.lower() in _CONFIDENCE_LEVELS:
                confidence = p.lower()
                break
        category = ""
        for p in parts[2:]:
            if p in CATEGORIES:
                category = p
                break
        results[term] = (is_term_text.lower() == "yes", category, confidence)
    return results


def classify_with_llm(
    terms: list[str], model: str = DEFAULT_MODEL, host: str | None = None, batch_size: int = 20
) -> dict[str, tuple[bool, str, str]]:
    expected = set(terms)
    classify_map: dict[str, tuple[bool, str, str]] = {}
    for i in range(0, len(terms), batch_size):
        batch = terms[i : i + batch_size]
        prompt = TERM_CLASSIFY_PROMPT_TEMPLATE.format(terms="\n".join(batch))
        try:
            raw = call_ollama(prompt, model=model, host=host)
        except Exception as exc:  # noqa: BLE001 - keep going on a single bad batch
            print(f"WARN: LLM call failed for batch starting at {i}: {exc}")
            continue
        classify_map.update(_parse_classify_response(raw, expected))
    return classify_map


def load_ontology(path: Path | str = DEFAULT_OUTPUT_PATH) -> dict[str, list[str]]:
    path = Path(path)
    if not path.is_file():
        return {cat: [] for cat in CATEGORIES}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for cat in CATEGORIES:
        data.setdefault(cat, [])
    return data


def save_ontology(ontology: dict[str, list[str]], path: Path | str = DEFAULT_OUTPUT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# domain_dict/medical_ontology.yaml -- Medical Ontology (research MVP, NOT for clinical use)\n"
        "#\n"
        "# 5-layer taxonomy (Disease/Symptom/Procedure/Drug/Body-Part) built from:\n"
        "#   - PriMock57 notes/*.json `highlights` + `presenting_complaint` (clinician-flagged, audio side)\n"
        "#   - IU X-ray report_text keyword matches against a curated chest X-ray finding list (image side)\n"
        "# Classified into categories by a local LLM (qwen2.5:7b-instruct-q4_k_m); only confidence=\"high\"\n"
        "# candidates are merged here (see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 5, Decisions).\n"
        "# Research/technical-demonstration use only; NOT validated for clinical decision support.\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(ontology, f, allow_unicode=True, sort_keys=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 5-category Medical Ontology from PriMock57 + IU X-ray candidates")
    parser.add_argument("--notes-dir", default=str(DEFAULT_NOTES_DIR))
    parser.add_argument("--iuxray-pairs", default=str(DEFAULT_IUXRAY_PAIRS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    audio_candidates = collect_audio_side_candidates(args.notes_dir)
    image_candidates = collect_image_side_candidates(args.iuxray_pairs)
    procedure_candidates = collect_procedure_candidates(args.notes_dir, args.iuxray_pairs)
    print(f"audio-side candidates: {len(audio_candidates)} unique phrases ({sum(audio_candidates.values())} total mentions)")
    print(f"image-side candidates: {len(image_candidates)} unique phrases ({sum(image_candidates.values())} total mentions)")
    print(f"procedure candidates: {len(procedure_candidates)} unique phrases ({sum(procedure_candidates.values())} total mentions)")

    all_candidates = {
        t: f
        for t, f in {**audio_candidates, **image_candidates, **procedure_candidates}.items()
        if f >= args.min_freq
    }
    terms = sorted(all_candidates.keys())
    print(f"classifying {len(terms)} candidates with {args.model}...")

    classify_map = classify_with_llm(terms, model=args.model)

    ontology = load_ontology(args.output)
    merged = 0
    rejected_low_confidence = 0
    for term in terms:
        is_term, category, confidence = classify_map.get(term, (False, "", "low"))
        if not is_term or category not in CATEGORIES:
            continue
        if confidence != "high":
            rejected_low_confidence += 1
            continue
        if term not in ontology[category]:
            ontology[category].append(term)
            merged += 1

    for cat in CATEGORIES:
        ontology[cat] = sorted(set(ontology[cat]))

    save_ontology(ontology, args.output)
    counts = {cat: len(ontology[cat]) for cat in CATEGORIES}
    print(f"OK: merged {merged} new terms (rejected {rejected_low_confidence} as low/medium confidence) -> {args.output}")
    print(f"category counts: {counts}")


if __name__ == "__main__":
    main()
