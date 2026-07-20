## Results summary

| Model | Family | Anomaly AUC | Anomaly AP | Gen. acc. | Frechet | Likelihood metric |
|---|---|---|---|---|---|---|
| Conditional VAE | latent-variable | 0.846 | 0.869 | 1.000 | 0.175 | 0.728 bpd (ELBO bound) |
| RealNVP flow | explicit likelihood | 0.983 | 0.985 | 1.000 | 0.235 | -1.486 bpd (exact) |
| Conditional DDPM | denoising | 0.972 | 0.977 | 1.000 | 0.157 | 0.103 denoise MSE |
| Mamba (I/Q AR) | autoregressive | 0.829 | 0.842 | 1.000 | 2.530 | 5.399 bpd (exact discrete) |

### Anomaly-detection AUC by corruption type

| Model | heavy_noise | freq_shift | timing_corrupt | protocol_mix |
|---|---|---|---|---|
| Conditional VAE | 0.869 | 0.945 | 0.751 | 0.819 |
| RealNVP flow | 0.951 | 1.000 | 0.998 | 0.984 |
| Conditional DDPM | 0.936 | 1.000 | 0.996 | 0.956 |
| Mamba (I/Q AR) | 0.912 | 0.987 | 0.731 | 0.688 |

### Model size and training time

| Model | Params | Train time (s) |
|---|---|---|
| Conditional VAE | 596,065 | 8.6 |
| RealNVP flow | 6,901,856 | 17.8 |
| Conditional DDPM | 928,801 | 36.5 |
| Mamba (I/Q AR) | 356,376 | 1040.1 |
