"""Phase 8, Step 20: cross-domain CLAIP retrieval sanity check -- the core
deliverable demonstrating CLAIP's domain generality (see
docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md TL;DR/Phase 8).

For each query term, the *same* frozen TextEncoder
(encoders_medical.TextEncoder, multilingual-e5-base) embeds the query
once; that raw embedding is then projected through TWO independently
trained `text_proj` heads -- one from the Audio-Text branch
(PriMock57, train_projection_pair.py --pair-names audio,text) and one
from the Image-Text branch (IU X-ray, --pair-names image,text) -- to
search each branch's own embedding space. A "hit" means the top-1 nearest
neighbour's own text (note text for audio, report_text for image)
contains the query term.

As with the bridge domain's eval_retrieval.py, THIS IS A QUALITATIVE
SANITY CHECK on two small, independently-sourced closed sets (57
consultations / IU X-ray report pairs), not a statistically validated
Recall@K benchmark. v0.1.1 finding: the two datasets cover genuinely
different clinical vocabularies (UK primary-care symptom phrasing vs. US
chest-radiology finding descriptions) -- e.g. "cough"/"fever"/"asthma"/
"chest pain"/"breathlessness" literally never appear in any of 2,972 IU
X-ray reports. Query terms are therefore NOT hand-picked but selected
mechanically via find_shared_queries() (terms verified to co-occur in
both corpora), per docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 8 /
Decisions.

Usage:
    python src_claip/eval_cross_domain_retrieval.py
    python src_claip/eval_cross_domain_retrieval.py --queries "pneumonia,hypertension"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from encoders_medical import TextEncoder
from train_projection_pair import PairProjectionHeads

DEFAULT_AUDIO_TEXT_DIR = Path("data/processed/audio_text/claip")
DEFAULT_IMAGE_TEXT_DIR = Path("data/processed/image_text/claip")
DEFAULT_ONTOLOGY_PATH = Path("domain_dict/medical_ontology.yaml")
DEFAULT_OUTPUT_PATH = Path("output/medical_claip_cross_domain_retrieval.json")
DEFAULT_QUERIES = "pneumonia,cough,fever,chest pain,emphysema,asthma,breathlessness"


def load_projection(claip_dir: Path, device: str) -> tuple[PairProjectionHeads, list[str]]:
    ckpt = torch.load(claip_dir / "projection.pt", map_location=device)
    model = PairProjectionHeads(ckpt["a_dim"], ckpt["b_dim"], proj_dim=ckpt["proj_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["pair_names"]


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def find_shared_queries(
    ontology_path: Path,
    audio_triples_path: Path,
    image_triples_path: Path,
    categories: tuple[str, ...] = ("Disease", "Symptom", "Body-Part", "Procedure", "Drug"),
    max_queries: int = 10,
) -> list[str]:
    """Mechanically select query terms that literally co-occur in BOTH
    branches' own source text (audio_text's note_text, image_text's
    report_text), sorted by combined frequency. See
    docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 8, Step 20: a
    hand-picked query guess list risks querying for terms that simply
    never occur in one of the two corpora (verified empirically in v0.1.1
    -- "cough"/"fever"/"asthma"/"chest pain"/"breathlessness" appear in
    zero of 2,972 IU X-ray reports, since radiology reports describe
    imaging findings, not patient-reported symptoms), which silently caps
    the achievable hit rate at 0% regardless of retrieval quality. This
    function instead checks actual presence, the same mechanical
    frequency-counting approach as term_extraction_medical.py."""
    with open(ontology_path, encoding="utf-8") as f:
        ontology = yaml.safe_load(f) or {}
    candidates = sorted({t for cat in categories for t in (ontology.get(cat) or [])})

    with open(audio_triples_path, encoding="utf-8") as f:
        audio_texts = [t.get("note_text", "") for t in json.load(f)]
    with open(image_triples_path, encoding="utf-8") as f:
        image_texts = [t.get("report_text", "") for t in json.load(f)]
    # dedupe identical note_text repeated across a consultation's utterances
    audio_texts = list(dict.fromkeys(audio_texts))

    scored: list[tuple[str, int, int]] = []
    for term in candidates:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        n_audio = sum(1 for t in audio_texts if pattern.search(t))
        n_image = sum(1 for t in image_texts if pattern.search(t))
        if n_audio > 0 and n_image > 0:
            scored.append((term, n_audio, n_image))

    scored.sort(key=lambda x: -(x[1] + x[2]))
    return [term for term, _, _ in scored[:max_queries]]


def run_eval(
    audio_text_dir: Path, image_text_dir: Path, queries: list[str], topk: int = 5
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = TextEncoder(device=device)

    audio_triples = json.loads((audio_text_dir / "triples.json").read_text(encoding="utf-8"))
    audio_emb = np.load(audio_text_dir / "audio_emb.npy")
    audio_model, audio_pair_names = load_projection(audio_text_dir, device)
    assert audio_pair_names == ["audio", "text"], audio_pair_names

    image_triples = json.loads((image_text_dir / "triples.json").read_text(encoding="utf-8"))
    image_emb = np.load(image_text_dir / "image_emb.npy")
    image_model, image_pair_names = load_projection(image_text_dir, device)
    assert image_pair_names == ["image", "text"], image_pair_names

    with torch.no_grad():
        audio_proj_all = F.normalize(audio_model.a_proj(torch.tensor(audio_emb, dtype=torch.float32, device=device)), dim=-1).cpu().numpy()
        image_proj_all = F.normalize(image_model.a_proj(torch.tensor(image_emb, dtype=torch.float32, device=device)), dim=-1).cpu().numpy()

    results = []
    audio_hits = 0
    image_hits = 0
    for query in queries:
        raw_text_emb = text_encoder.encode_text(query)
        raw_t = torch.tensor(raw_text_emb, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            audio_query_vec = F.normalize(audio_model.b_proj(raw_t), dim=-1).squeeze(0).cpu().numpy()
            image_query_vec = F.normalize(image_model.b_proj(raw_t), dim=-1).squeeze(0).cpu().numpy()

        a_idx, a_sims = cosine_topk(audio_query_vec, audio_proj_all, k=topk)
        i_idx, i_sims = cosine_topk(image_query_vec, image_proj_all, k=topk)

        top_audio = audio_triples[int(a_idx[0])]
        top_image = image_triples[int(i_idx[0])]

        audio_hit = query.lower() in (top_audio.get("note_text", "") or "").lower()
        image_hit = query.lower() in (top_image.get("report_text", "") or "").lower()
        audio_hits += int(audio_hit)
        image_hits += int(image_hit)

        results.append(
            {
                "query": query,
                "audio_hit": audio_hit,
                "audio_top_consultation_id": top_audio.get("consultation_id"),
                "audio_top_sim": float(a_sims[0]),
                "audio_top_note_text": (top_audio.get("note_text", "") or "")[:200],
                "image_hit": image_hit,
                "image_top_image_id": top_image.get("image_id"),
                "image_top_sim": float(i_sims[0]),
                "image_top_report_text": (top_image.get("report_text", "") or "")[:200],
            }
        )

    text_encoder.unload()
    return {
        "note": (
            "Qualitative cross-domain sanity check: two independently trained 2-modality CLAIP "
            "branches (Audio-Text on PriMock57, Image-Text on IU X-ray) sharing only the same "
            "frozen multilingual-e5-base TextEncoder. NOT a statistically validated Recall@K "
            "benchmark; the two source datasets cover different clinical contexts (UK primary "
            "care vs. US chest radiology), so some queries may legitimately show low overlap. "
            "See docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Phase 8 / Further Considerations."
        ),
        "audio_hit_rate": f"{audio_hits}/{len(queries)}",
        "image_hit_rate": f"{image_hits}/{len(queries)}",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain CLAIP retrieval sanity check (Audio-Text + Image-Text branches)")
    parser.add_argument("--audio-text-dir", default=str(DEFAULT_AUDIO_TEXT_DIR))
    parser.add_argument("--image-text-dir", default=str(DEFAULT_IMAGE_TEXT_DIR))
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    parser.add_argument(
        "--queries", default=None,
        help="Comma-separated query terms. If omitted, queries are selected mechanically via "
             "find_shared_queries() (terms that literally co-occur in both branches' own source "
             "text) -- see module docstring on why a hand-picked guess list is unreliable.",
    )
    parser.add_argument("--max-queries", type=int, default=10, help="Cap on mechanically-selected queries (ignored if --queries given)")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    if args.queries:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    else:
        queries = find_shared_queries(
            Path(args.ontology),
            Path(args.audio_text_dir) / "triples.json",
            Path(args.image_text_dir) / "triples.json",
            max_queries=args.max_queries,
        )
        print(f"mechanically selected {len(queries)} shared queries: {queries}")
    report = run_eval(Path(args.audio_text_dir), Path(args.image_text_dir), queries, topk=args.topk)

    print("=== Cross-domain CLAIP retrieval sanity check (qualitative) ===")
    for r in report["results"]:
        print(
            f"  query={r['query']!r} audio_hit={r['audio_hit']} image_hit={r['image_hit']} "
            f"audio_top_note={r['audio_top_note_text'][:50]!r} image_top_report={r['image_top_report_text'][:50]!r}"
        )
    print(f"audio hit rate: {report['audio_hit_rate']}  image hit rate: {report['image_hit_rate']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
