"""v0.2.0: true 3-modality CLAIP projection-head trainer for the
single-domain (image, audio, text) triples (build_triples_single_domain.py).
Ported from the bridge domain's src_claip/train_claip_projection.py
(ProjectionHeads + weighted 3-pairwise InfoNCE loss), restoring the
original CLAIP architecture that v0.1.x's two disjoint 2-modality branches
had to depart from (no public dataset paired audio+image+text for the same
medical case -- see docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md v0.1.1
addendum). `info_nce_pair()` is copied verbatim from both
train_claip_projection.py and train_projection_pair.py.

Usage:
    python src_claip/train_triple_projection.py --triples-dir data/processed/image_text/claip_v0.2.0
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
DEFAULT_EPOCHS = 500
DEFAULT_LR = 1e-3
DEFAULT_TAU = 0.07
DEFAULT_SEED = 42


class TripleProjectionHeads(nn.Module):
    """Three independent linear projections (no shared weights), same
    design as the bridge domain's ProjectionHeads."""

    def __init__(self, image_dim: int, audio_dim: int, text_dim: int, proj_dim: int = DEFAULT_PROJ_DIM):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

    def forward(self, image_emb: torch.Tensor, audio_emb: torch.Tensor, text_emb: torch.Tensor):
        img = F.normalize(self.image_proj(image_emb), dim=-1)
        aud = F.normalize(self.audio_proj(audio_emb), dim=-1)
        txt = F.normalize(self.text_proj(text_emb), dim=-1)
        return img, aud, txt


def info_nce_pair(a: torch.Tensor, b: torch.Tensor, tau: float = DEFAULT_TAU) -> torch.Tensor:
    logits = a @ b.T / tau
    labels = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train(
    image_emb: np.ndarray,
    audio_emb: np.ndarray,
    text_emb: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    proj_dim: int = DEFAULT_PROJ_DIM,
    tau: float = DEFAULT_TAU,
    w_img_txt: float = 1.0,
    w_aud_txt: float = 1.0,
    w_img_aud: float = 1.0,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> tuple[TripleProjectionHeads, list[dict]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    image_t = torch.tensor(image_emb, dtype=torch.float32, device=device)
    audio_t = torch.tensor(audio_emb, dtype=torch.float32, device=device)
    text_t = torch.tensor(text_emb, dtype=torch.float32, device=device)

    model = TripleProjectionHeads(image_t.shape[1], audio_t.shape[1], text_t.shape[1], proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    best_loss = float("inf")
    best_state: dict | None = None
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        img, aud, txt = model(image_t, audio_t, text_t)
        loss_img_txt = info_nce_pair(img, txt, tau)
        loss_aud_txt = info_nce_pair(aud, txt, tau)
        loss_img_aud = info_nce_pair(img, aud, tau)
        loss = w_img_txt * loss_img_txt + w_aud_txt * loss_aud_txt + w_img_aud * loss_img_aud
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        history.append(
            {
                "total": loss_value,
                "img_txt": loss_img_txt.item(),
                "aud_txt": loss_aud_txt.item(),
                "img_aud": loss_img_aud.item(),
            }
        )
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1}/{epochs} loss={loss_value:.4f} "
                f"(img-txt={loss_img_txt.item():.4f} aud-txt={loss_aud_txt.item():.4f} img-aud={loss_img_aud.item():.4f})"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best checkpoint: loss={best_loss:.4f} (final epoch loss={history[-1]['total']:.4f})")
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single-domain 3-modality CLAIP projection heads")
    parser.add_argument("--triples-dir", required=True, help="Directory holding image_emb.npy/audio_emb.npy/text_emb.npy/triples.json")
    parser.add_argument("--output-dir", default=None, help="Defaults to --triples-dir (in-place)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--proj-dim", type=int, default=DEFAULT_PROJ_DIM)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--w-img-txt", type=float, default=1.0)
    parser.add_argument("--w-aud-txt", type=float, default=1.0)
    parser.add_argument("--w-img-aud", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    triples_dir = Path(args.triples_dir)
    output_dir = Path(args.output_dir) if args.output_dir else triples_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_emb = np.load(triples_dir / "image_emb.npy")
    audio_emb = np.load(triples_dir / "audio_emb.npy")
    text_emb = np.load(triples_dir / "text_emb.npy")
    if len(image_emb) < 2:
        raise RuntimeError(f"Need at least 2 triples for in-batch-negative contrastive training, found {len(image_emb)}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train(
        image_emb, audio_emb, text_emb, epochs=args.epochs, lr=args.lr, proj_dim=args.proj_dim, tau=args.tau,
        w_img_txt=args.w_img_txt, w_aud_txt=args.w_aud_txt, w_img_aud=args.w_img_aud, seed=args.seed, device=device,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_dim": image_emb.shape[1],
            "audio_dim": audio_emb.shape[1],
            "text_dim": text_emb.shape[1],
            "proj_dim": args.proj_dim,
        },
        output_dir / "projection.pt",
    )

    model.eval()
    with torch.no_grad():
        img, aud, txt = model(
            torch.tensor(image_emb, dtype=torch.float32, device=device),
            torch.tensor(audio_emb, dtype=torch.float32, device=device),
            torch.tensor(text_emb, dtype=torch.float32, device=device),
        )
    np.save(output_dir / "image_proj.npy", img.cpu().numpy())
    np.save(output_dir / "audio_proj.npy", aud.cpu().numpy())
    np.save(output_dir / "text_proj.npy", txt.cpu().numpy())
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "n_triples": len(image_emb), "seed": args.seed}, f)

    final = history[-1]
    print(
        f"OK: trained 3-modality projection heads on {len(image_emb)} triples, "
        f"final total loss={final['total']:.4f} (first={history[0]['total']:.4f}, best={min(h['total'] for h in history):.4f}) "
        f"-> {output_dir}"
    )


if __name__ == "__main__":
    main()
