"""Generate figures and a results-summary table for the CLAIP extension paper
from the actual meet_mp4_claip v0.1.0/v0.2.0 artifacts (output/*.md reports,
data/processed/<stem>/claip{,_v0.1.0}/training_history.json,
retrieval_eval.json). Run once from the meet_mp4_claip project root with the
.venv-asr (or any Python with matplotlib) interpreter.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "gmeet_261020_WorkExternalReview"
FIGDIR = Path(__file__).resolve().parent / "figures"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_frame_density() -> None:
    """Fig2: kept frames and contrastive-triple count, v0.1.0 (interval)
    vs v0.2.0 (segment-driven). Source: evaluate_frame_extraction_methods.py
    and build_triples.py console output, recorded in
    output/gmeet_261020_WorkExternalReview_frame_extraction_method_comparison.md
    and output/gmeet_261020_WorkExternalReview_claip_eval_report.md."""
    labels = ["v0.1.0\n(10s interval)", "v0.2.0\n(segment-driven)"]
    kept_frames = [92, 264]
    candidates = [55, 93]
    triples = [90, 961]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, values, title, color in zip(
        axes,
        [kept_frames, candidates, triples],
        ["Kept frames\n(post phash-dedup)", "Candidate terms\n(fixed 183-term ref. vocab)",
         "Contrastive triples\n(image, audio, text)"],
        ["#4C72B0", "#55A868", "#C44E52"],
    ):
        bars = ax.bar(labels, values, color=color)
        ax.bar_label(bars, fmt="%d", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(values) * 1.3)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Frame-extraction density: v0.1.0 (interval) vs v0.2.0 (segment-driven)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(FIGDIR / "Fig2_frame_density.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_training_curves() -> None:
    """Fig3: InfoNCE training loss curves, v0.1.0 (90 triples, unweighted) vs
    v0.2.0 (961 triples, unweighted) vs v0.2.1 (961 triples, weighted per
    Equation~\\ref{eq:weighted-infonce}, w_img_txt=0.75, w_aud_txt=1.50,
    w_img_aud=0.75), all with best-checkpoint tracking."""
    v010 = load_json(PROCESSED / "claip_v0.1.0" / "training_history.json")
    v020 = load_json(PROCESSED / "claip" / "training_history.json")
    v021_path = PROCESSED / "claip_v0.2.1_weighted" / "training_history.json"
    v021 = load_json(v021_path) if v021_path.exists() else None

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(range(1, len(v010["loss"]) + 1), v010["loss"], label=f"v0.1.0 (N={v010['n_triples']} triples)",
            color="#4C72B0", linewidth=1.5)
    ax.plot(range(1, len(v020["loss"]) + 1), v020["loss"], label=f"v0.2.0 (N={v020['n_triples']} triples, unweighted)",
            color="#C44E52", linewidth=1.5)
    if v021 is not None:
        w = v021.get("weights", {})
        label = (
            f"v0.2.1 (N={v021['n_triples']} triples, weighted "
            f"w1={w.get('w_img_txt', '?')}, w2={w.get('w_aud_txt', '?')}, w3={w.get('w_img_aud', '?')})"
        )
        ax.plot(range(1, len(v021["loss"]) + 1), v021["loss"], label=label, color="#55A868", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Symmetric InfoNCE loss (sum over 3 modality pairs)", fontsize=9)
    ax.set_title("CLAIP projection-head training loss", fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    plt.savefig(FIGDIR / "Fig3_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_retrieval_hitrate() -> None:
    """Fig4: per-query retrieval hit (v0.1.0 vs v0.2.0), from the qualitative
    sanity check (7 known high-confidence domain terms from v0.2.1). Query
    labels are romanized (the default matplotlib font cannot render
    Japanese glyphs), matching the paper's convention of romanizing
    Japanese terms in italics."""
    romanized = {
        "当て板": "ateita", "アーチリブ": "aachi-rib", "下フランジ": "shita-flange",
        "リベット": "rivet", "中詰め": "nakazume", "構造形式": "kouzou-keishiki",
        "補修方法": "hoshuu-houhou",
    }
    v010 = load_json(PROCESSED / "claip_v0.1.0" / "retrieval_eval.json")
    v020 = load_json(PROCESSED / "claip" / "retrieval_eval.json")
    queries = [r["query"] for r in v020["results"]]
    query_labels = [romanized.get(q, q) for q in queries]
    v010_hits = {r["query"]: r["hit"] for r in v010["results"]}
    v010_vals = [1 if v010_hits.get(q) else 0 for q in queries]
    v020_vals = [1 if r["hit"] else 0 for r in v020["results"]]

    x = range(len(queries))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar([i - width / 2 for i in x], v010_vals, width, label="v0.1.0 (frame-centric, 90 triples)", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], v020_vals, width, label="v0.2.0 (segment-centric, 961 triples)", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(query_labels, rotation=30, ha="right", fontsize=8, style="italic")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["miss", "hit"], fontsize=8)
    ax.set_title(
        f"Cross-modal retrieval sanity check: v0.1.0 {sum(v010_vals)}/7 vs v0.2.0 {sum(v020_vals)}/7",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    plt.savefig(FIGDIR / "Fig4_retrieval_hitrate.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_frame_density()
    fig_training_curves()
    fig_retrieval_hitrate()
    print("Figures written to", FIGDIR)


if __name__ == "__main__":
    main()
