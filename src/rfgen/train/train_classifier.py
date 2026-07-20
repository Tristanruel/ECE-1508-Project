from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rfgen.data import make_loaders, load_split
from rfgen.models import SpectrogramCNN
from rfgen.utils import count_params, ensure_dir, get_device, load_config, save_json, set_seed


@torch.no_grad()
def _accuracy(model, loader, device) -> float:
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n += x.shape[0]
    return correct / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device()

    train_loader, val_loader = make_loaders(cfg)
    model = SpectrogramCNN(
        n_classes=len(cfg["signal"]["classes"]),
        base_ch=int(cfg["models"]["classifier"]["base_ch"]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = int(cfg["train"]["epochs_classifier"])
    run_dir = ensure_dir(Path(cfg["train"]["runs_dir"]) / "classifier")
    print(f"[train:classifier] params={count_params(model):,} device={device}")

    best_acc, hist = 0.0, {"train_loss": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        run, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"clf ep{epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * x.shape[0]
            n += x.shape[0]
        acc = _accuracy(model, val_loader, device)
        hist["train_loss"].append(run / max(n, 1))
        hist["val_acc"].append(acc)
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "val_acc": acc}, run_dir / "best.pt")


    from rfgen.train.common import load_best
    load_best(model, cfg, "classifier", device)
    test = load_split(cfg, "test")
    from torch.utils.data import DataLoader
    test_acc = _accuracy(model, DataLoader(test, batch_size=256), device)
    summary = {
        "params": count_params(model), "best_val_acc": best_acc,
        "test_acc": test_acc, "seconds": round(time.time() - t0, 1), "history": hist,
    }
    save_json(summary, run_dir / "history.json")
    print(f"[train:classifier] best_val_acc={best_acc:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
