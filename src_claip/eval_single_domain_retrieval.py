"""v0.2.0: single-domain 3-modality CLAIP retrieval sanity check --
replaces v0.1.x's eval_cross_domain_retrieval.py (two disjoint branches)
now that image/audio/text all come from the same chest-radiology domain
and the same case (build_triples_single_domain.py + train_triple_projection.py).

For each query term, the shared frozen TextEncoder embeds the query once;
the projection is applied via the SAME trained model's `text_proj`, then
searched against `image_proj`/`audio_proj` (text->image, text->audio), and
additionally `image_proj` is searched directly against `audio_proj`
(image->audio), exercising all 3 pairwise directions the bridge domain's
eval_retrieval.py also qualitatively checked.

Still a qualitative sanity check on one dataset's closed set (1,000 IU
X-ray cases, see bridge domain's eval_retrieval.py docstring for the same
caveat), NOT a statistically validated Recall@K benchmark -- but unlike
v0.1.x, query term selection no longer needs cross-corpus-overlap
verification (find_shared_queries()): any term that appears in the IU
X-ray report corpus is automatically "shared" across all 3 modalities of
the same case.

Usage:
    python src_claip/eval_single_domain_retrieval.py
    python src_claip/eval_single_domain_retrieval.py --queries "pneumonia,cardiomegaly,effusion"
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
from train_triple_projection import TripleProjectionHeads

DEFAULT_TRIPLES_DIR = Path("data/processed/image_text/claip_v0.2.0")
DEFAULT_ONTOLOGY_PATH = Path("domain_dict/medical_ontology.yaml")
DEFAULT_OUTPUT_PATH = Path("output/medical_claip_v0.2.0_retrieval.json")
DEFAULT_TOPK = 5


def load_projection(triples_dir: Path, device: str) -> TripleProjectionHeads:
    ckpt = torch.load(triples_dir / "projection.pt", map_location=device)
    model = TripleProjectionHeads(ckpt["image_dim"], ckpt["audio_dim"], ckpt["text_dim"], proj_dim=ckpt["proj_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def default_queries(ontology_path: Path, pairs_path: Path, max_queries: int = 10) -> list[str]:
    """Any ontology Disease/Symptom/Body-Part term that actually occurs in
    the report corpus (no cross-corpus check needed here, since all 3
    modalities share one corpus by construction -- contrast with v0.1.x's
    find_shared_queries())."""
    with open(ontology_path, encoding="utf-8") as f:
        ontology = yaml.safe_load(f) or {}
    candidates = sorted({t for cat in ("Disease", "Symptom", "Body-Part") for t in (ontology.get(cat) or [])})

    with open(pairs_path, encoding="utf-8") as f:
        texts = [t.get("report_text", "") for t in json.load(f)]

    scored = []
    for term in candidates:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        n = sum(1 for t in texts if pattern.search(t))
        if n > 0:
            scored.append((term, n))
    scored.sort(key=lambda x: -x[1])
    return [t for t, _ in scored[:max_queries]]


def global_self_retrieval_accuracy(a_proj: np.ndarray, b_proj: np.ndarray, topk: int = 1) -> float:
    """Over ALL N triples (not just a handful of query terms): for each
    case i, does a_proj[i] retrieve b_proj[i] among its top-k nearest
    neighbours in b_proj? This is the standard closed-set Recall@k
    self-retrieval check for a triple-based contrastive model -- a much
    larger-sample, query-independent complement to the per-query hit-rate
    checks below."""
    a_norm = a_proj / (np.linalg.norm(a_proj, axis=1, keepdims=True) + 1e-8)
    b_norm = b_proj / (np.linalg.norm(b_proj, axis=1, keepdims=True) + 1e-8)
    sims = a_norm @ b_norm.T  # (N, N)
    topk_idx = np.argsort(-sims, axis=1)[:, :topk]
    hits = sum(1 for i in range(len(a_proj)) if i in topk_idx[i])
    return hits / len(a_proj)


def run_eval(triples_dir: Path, queries: list[str], topk: int = DEFAULT_TOPK) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    triples = json.loads((triples_dir / "triples.json").read_text(encoding="utf-8"))
    image_emb = np.load(triples_dir / "image_emb.npy")
    audio_emb = np.load(triples_dir / "audio_emb.npy")

    model = load_projection(triples_dir, device)
    text_encoder = TextEncoder(device=device)

    with torch.no_grad():
        image_proj_all = F.normalize(model.image_proj(torch.tensor(image_emb, dtype=torch.float32, device=device)), dim=-1).cpu().numpy()
        audio_proj_all = F.normalize(model.audio_proj(torch.tensor(audio_emb, dtype=torch.float32, device=device)), dim=-1).cpu().numpy()

    results = []
    txt_img_hits = txt_aud_hits = img_aud_hits = 0
    for query in queries:
        raw_text_emb = text_encoder.encode_text(query)
        with torch.no_grad():
            text_vec = model.text_proj(torch.tensor(raw_text_emb, dtype=torch.float32, device=device).unsqueeze(0))
            text_vec = F.normalize(text_vec, dim=-1).squeeze(0).cpu().numpy()

        img_idx, img_sims = cosine_topk(text_vec, image_proj_all, k=topk)
        aud_idx, aud_sims = cosine_topk(text_vec, audio_proj_all, k=topk)

        top_image = triples[int(img_idx[0])]
        top_audio = triples[int(aud_idx[0])]
        txt_img_hit = query.lower() in (top_image.get("report_text", "") or "").lower()
        txt_aud_hit = query.lower() in (top_audio.get("report_text", "") or "").lower()
        # image->audio: does querying with THIS case's own image retrieve
        # its OWN case's audio at top-1 (a correct cross-modal match for a
        # same-case triple, the InfoNCE positive-pair objective itself)?
        query_case_idx = next((t["row"] for t in triples if query.lower() in (t.get("report_text", "") or "").lower()), None)
        img_aud_hit = False
        if query_case_idx is not None:
            aud_idx2, _ = cosine_topk(image_proj_all[query_case_idx], audio_proj_all, k=1)
            img_aud_hit = int(aud_idx2[0]) == query_case_idx

        txt_img_hits += int(txt_img_hit)
        txt_aud_hits += int(txt_aud_hit)
        img_aud_hits += int(img_aud_hit)

        results.append(
            {
                "query": query,
                "text_to_image_hit": txt_img_hit,
                "text_to_image_top_report": (top_image.get("report_text", "") or "")[:150],
                "text_to_audio_hit": txt_aud_hit,
                "text_to_audio_top_report": (top_audio.get("report_text", "") or "")[:150],
                "image_to_audio_same_case_hit": img_aud_hit,
            }
        )

    text_encoder.unload()

    global_img_to_aud_r1 = global_self_retrieval_accuracy(image_proj_all, audio_proj_all, topk=1)
    global_aud_to_img_r1 = global_self_retrieval_accuracy(audio_proj_all, image_proj_all, topk=1)
    global_img_to_aud_r5 = global_self_retrieval_accuracy(image_proj_all, audio_proj_all, topk=5)

    return {
        "note": (
            "Qualitative sanity check on one single-domain (chest radiology) dataset's closed set "
            "(1,000 IU X-ray cases with TTS-synthesized narration audio of their own report). "
            "The per-query hit-rate numbers below are a small-sample qualitative check (NOT a "
            "statistically validated Recall@K benchmark); the global_* numbers are a query-"
            "independent closed-set Recall@k computed over all N triples, a much larger sample. "
            "See docs/Plan_v0.2.0_Medical_Ontology_CLAIP.md."
        ),
        "text_to_image_hit_rate": f"{txt_img_hits}/{len(queries)}",
        "text_to_audio_hit_rate": f"{txt_aud_hits}/{len(queries)}",
        "image_to_audio_same_case_hit_rate": f"{img_aud_hits}/{len(queries)}",
        "global_image_to_audio_recall@1": global_img_to_aud_r1,
        "global_audio_to_image_recall@1": global_aud_to_img_r1,
        "global_image_to_audio_recall@5": global_img_to_aud_r5,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-domain 3-modality CLAIP retrieval sanity check")
    parser.add_argument("--triples-dir", default=str(DEFAULT_TRIPLES_DIR))
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    parser.add_argument("--queries", default=None, help="Comma-separated query terms (default: mechanically selected)")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    triples_dir = Path(args.triples_dir)
    if args.queries:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    else:
        queries = default_queries(Path(args.ontology), triples_dir / "triples.json", max_queries=args.max_queries)
        print(f"mechanically selected {len(queries)} queries: {queries}")

    report = run_eval(triples_dir, queries, topk=args.topk)

    print("=== Single-domain CLAIP retrieval sanity check (qualitative) ===")
    for r in report["results"]:
        print(
            f"  query={r['query']!r} text->image_hit={r['text_to_image_hit']} "
            f"text->audio_hit={r['text_to_audio_hit']} image->audio(own case)_hit={r['image_to_audio_same_case_hit']}"
        )
    print(
        f"text->image hit rate: {report['text_to_image_hit_rate']}  "
        f"text->audio hit rate: {report['text_to_audio_hit_rate']}  "
        f"image->audio(own case) hit rate: {report['image_to_audio_same_case_hit_rate']}"
    )
    print(
        f"global image->audio Recall@1: {report['global_image_to_audio_recall@1']:.4f}  "
        f"global audio->image Recall@1: {report['global_audio_to_image_recall@1']:.4f}  "
        f"global image->audio Recall@5: {report['global_image_to_audio_recall@5']:.4f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
