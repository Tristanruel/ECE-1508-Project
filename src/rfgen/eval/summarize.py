from __future__ import annotations

import argparse

from rfgen.utils import load_config, load_json, resolve_path

TAGS = ["vae", "flow", "diffusion", "mamba"]
LABELS = {"vae": "Conditional VAE", "flow": "RealNVP flow", "diffusion": "Conditional DDPM",
          "mamba": "Mamba (I/Q AR)"}
FAMILY = {"vae": "latent-variable", "flow": "explicit likelihood", "diffusion": "denoising",
          "mamba": "autoregressive"}


def build_summary(cfg: dict) -> str:
    an = load_json(f"{cfg['results_dir']}/anomaly_metrics.json")
    gen = load_json(f"{cfg['results_dir']}/generation_metrics.json")
    lines = []
    lines.append("## Results summary\n")


    lines.append("| Model | Family | Anomaly AUC | Anomaly AP | Gen. acc. | Frechet | Likelihood metric |")
    lines.append("|---|---|---|---|---|---|---|")
    lik = gen["likelihood"]
    lik_str = {
        "flow": f"{lik['flow']['test_bpd']:.3f} bpd (exact)",
        "vae": f"{lik['vae']['test_bpd_bound']:.3f} bpd (ELBO bound)",
        "diffusion": f"{lik['diffusion']['test_denoise_mse']:.3f} denoise MSE",
        "mamba": f"{lik['mamba']['test_bpd']:.3f} bpd (exact discrete)",
    }
    for t in TAGS:
        a = an["per_model"][t]; g = gen["per_model"][t]
        lines.append(
            f"| {LABELS[t]} | {FAMILY[t]} | {a['auc']:.3f} | {a['ap']:.3f} | "
            f"{g['gen_accuracy']:.3f} | {g['frechet']:.3f} | {lik_str[t]} |"
        )


    types = an["anomaly_types"]
    lines.append("\n### Anomaly-detection AUC by corruption type\n")
    lines.append("| Model | " + " | ".join(types) + " |")
    lines.append("|---" * (len(types) + 1) + "|")
    for t in TAGS:
        pt = an["per_model"][t]["per_type_auc"]
        lines.append(f"| {LABELS[t]} | " + " | ".join(f"{pt[k]:.3f}" for k in types) + " |")


    lines.append("\n### Model size and training time\n")
    lines.append("| Model | Params | Train time (s) |")
    lines.append("|---|---|---|")
    for t in TAGS:
        h = load_json(f"{cfg['train']['runs_dir']}/{t}/history.json")
        lines.append(f"| {LABELS[t]} | {h['params']:,} | {h['seconds']:.1f} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    md = build_summary(cfg)
    out = resolve_path(f"{cfg['results_dir']}/summary.md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[summarize] wrote {out}")


if __name__ == "__main__":
    main()
