"""Phase 7, Step 19: generalized 2-modality CLAIP projection-head trainer.

Medical-domain analogue of the bridge CLAIP pipeline's
src_claip/train_claip_projection.py, simplified from 3 frozen encoders
(image/audio/text) + 3-pairwise-loss `ProjectionHeads` down to 2 frozen
encoders + a single `info_nce_pair()` loss term, since neither medical
branch has all 3 modalities at once (see
docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md Decisions: no public dataset
pairs audio+image+text for the same medical encounter). `info_nce_pair()`
itself is copied verbatim from train_claip_projection.py.

Used twice, independently, with --pair-names audio,text (PriMock57) and
--pair-names image,text (IU X-ray) -- both branches share the same
`proj_dim`/`tau` defaults so their outputs are comparable, and both use the
*same* frozen TextEncoder (encoders_medical.TextEncoder) as their text
input, which is what lets eval_cross_domain_retrieval.py query both
branches with one shared vocabulary.

Usage:
    python src_claip/train_projection_pair.py \\
        --triples data/processed/audio_text/claip/triples.json \\
        --emb-dir data/processed/audio_text/claip --pair-names audio,text \\
        --output-dir data/processed/audio_text/claip
    python src_claip/train_projection_pair.py \\
        --triples data/processed/image_text/claip/triples.json \\
        --emb-dir data/processed/image_text/claip --pair-names image,text \\
        --output-dir data/processed/image_text/claip
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PROJ_DIM = 256
DEFAULT_EPOCHS = 300
DEFAULT_LR = 1e-3
DEFAULT_TAU = 0.07
DEFAULT_SEED = 42


class PairProjectionHeads(nn.Module):
    """Two independent linear projections (no shared weights) mapping each
    frozen encoder's native dimensionality into a common `proj_dim`-d
    space, L2-normalized so dot products are cosine similarities. Named
    `a_proj`/`b_proj` generically (not image/audio/text) since the same
    class serves either modality pair."""

    def __init__(self, a_dim: int, b_dim: int, proj_dim: int = DEFAULT_PROJ_DIM):
        super().__init__()
        self.a_proj = nn.Linear(a_dim, proj_dim)
        self.b_proj = nn.Linear(b_dim, proj_dim)

    def forward(self, a_emb: torch.Tensor, b_emb: torch.Tensor):
        a = F.normalize(self.a_proj(a_emb), dim=-1)
        b = F.normalize(self.b_proj(b_emb), dim=-1)
        return a, b


def info_nce_pair(a: torch.Tensor, b: torch.Tensor, tau: float = DEFAULT_TAU) -> torch.Tensor:
    """Symmetric InfoNCE between two L2-normalized, row-aligned batches
    (row i of `a` and row i of `b` are a positive pair; all other rows in
    the batch are negatives). Copied verbatim from
    src_claip/train_claip_projection.py."""
    logits = a @ b.T / tau
    labels = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train(
    a_emb: np.ndarray,
    b_emb: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    proj_dim: int = DEFAULT_PROJ_DIM,
    tau: float = DEFAULT_TAU,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> tuple[PairProjectionHeads, list[float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    a_t = torch.tensor(a_emb, dtype=torch.float32, device=device)
    b_t = torch.tensor(b_emb, dtype=torch.float32, device=device)

    model = PairProjectionHeads(a_t.shape[1], b_t.shape[1], proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    best_loss = float("inf")
    best_state: dict | None = None
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        a_proj, b_proj = model(a_t, b_t)
        loss = info_nce_pair(a_proj, b_proj, tau)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        history.append(loss_value)
        # Track the best-loss checkpoint rather than blindly using the
        # final epoch (full-batch Adam without LR scheduling can spike
        # late in training; see train_claip_projection.py's matching note).
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs} loss={loss_value:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best checkpoint: loss={best_loss:.4f} (final epoch loss={history[-1]:.4f})")
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2-modality CLAIP projection head pair")
    parser.add_argument("--triples", required=True, help="Path to triples.json (for sanity-checking row count)")
    parser.add_argument("--emb-dir", required=True, help="Directory holding <a>_emb.npy / <b>_emb.npy")
    parser.add_argument("--pair-names", required=True, help="Comma-separated embedding file name prefixes, e.g. audio,text or image,text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--proj-dim", type=int, default=DEFAULT_PROJ_DIM)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    name_a, name_b = [n.strip() for n in args.pair_names.split(",")]
    emb_dir = Path(args.emb_dir)
    a_emb = np.load(emb_dir / f"{name_a}_emb.npy")
    b_emb = np.load(emb_dir / f"{name_b}_emb.npy")

    with open(args.triples, encoding="utf-8") as f:
        n_triples = len(json.load(f))
    if len(a_emb) != n_triples or len(b_emb) != n_triples:
        print(f"WARN: triples.json has {n_triples} rows but embeddings have {len(a_emb)}/{len(b_emb)} rows")
    if len(a_emb) < 2:
        raise RuntimeError(f"Need at least 2 triples for in-batch-negative contrastive training, found {len(a_emb)}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train(
        a_emb, b_emb, epochs=args.epochs, lr=args.lr, proj_dim=args.proj_dim, tau=args.tau, seed=args.seed, device=device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "a_dim": a_emb.shape[1],
            "b_dim": b_emb.shape[1],
            "proj_dim": args.proj_dim,
            "pair_names": [name_a, name_b],
        },
        output_dir / "projection.pt",
    )

    model.eval()
    with torch.no_grad():
        a_proj, b_proj = model(
            torch.tensor(a_emb, dtype=torch.float32, device=device),
            torch.tensor(b_emb, dtype=torch.float32, device=device),
        )
    np.save(output_dir / f"{name_a}_proj.npy", a_proj.cpu().numpy())
    np.save(output_dir / f"{name_b}_proj.npy", b_proj.cpu().numpy())
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"loss": history, "n_triples": len(a_emb), "pair_names": [name_a, name_b], "seed": args.seed}, f)

    print(
        f"OK: trained {name_a}<->{name_b} projection heads on {len(a_emb)} triples, "
        f"final loss={history[-1]:.4f} (first={history[0]:.4f}, best={min(history):.4f}) -> {output_dir}"
    )


if __name__ == "__main__":
    main()
