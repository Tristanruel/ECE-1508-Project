from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import tqdm

from rfgen.data import make_loaders
from rfgen.utils import count_params, ensure_dir, get_device, save_json, set_seed


@torch.no_grad()
def _val_loss(model, val_loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        loss, _ = model.loss(x, y)
        total += float(loss) * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def train_generative(model, cfg: dict, epochs: int, tag: str, loaders=None) -> dict:
    set_seed(int(cfg["seed"]))
    device = get_device()
    model = model.to(device)
    train_loader, val_loader = loaders if loaders is not None else make_loaders(cfg)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    run_dir = ensure_dir(Path(cfg["train"]["runs_dir"]) / tag)
    history = {"train_loss": [], "val_loss": [], "extra": []}
    best_val = float("inf")
    n_params = count_params(model)
    print(f"[train:{tag}] params={n_params:,} device={device} epochs={epochs}")

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        run, n, last = 0.0, 0, {}
        for x, y in tqdm(train_loader, desc=f"{tag} ep{epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss, logd = model.loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * x.shape[0]
            n += x.shape[0]
            last = logd
        tr = run / max(n, 1)
        va = _val_loss(model, val_loader, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["extra"].append(last)
        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": va}, run_dir / "best.pt")
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  ep{epoch+1:>3}  train={tr:.4f}  val={va:.4f}  {last}")

    elapsed = time.time() - t0
    torch.save({"model": model.state_dict(), "epoch": epochs - 1, "val": va}, run_dir / "last.pt")
    summary = {
        "tag": tag, "params": n_params, "epochs": epochs,
        "best_val": best_val, "final_val": va,
        "seconds": round(elapsed, 1), "device": str(device),
        "history": history,
    }
    save_json(summary, run_dir / "history.json")
    print(f"[train:{tag}] done best_val={best_val:.4f} in {elapsed:.1f}s -> {run_dir}")
    return summary


def load_best(model, cfg: dict, tag: str, device=None):
    device = device or get_device()
    ckpt = Path(cfg["train"]["runs_dir"]) / tag / "best.pt"
    from rfgen.utils import resolve_path
    state = torch.load(resolve_path(ckpt), map_location=device)
    model.load_state_dict(state["model"])
    return model.to(device).eval()
