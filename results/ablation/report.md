# Ablation results

Dataset: NJUD (sample), epochs: 3, seed: 42

| Variant | Fusion | Saliency | Train loss | AP@0.5 | Precision | Recall | F1 | Inference (ms) |
|---|---|---|---|---|---|---|---|---|
| baseline_no_fusion | none | zero | 0.1800 | 0.5836 | 0.3889 | 0.7000 | 0.5000 | 5699 |
| fusion_plain | plain | gt | 0.2091 | 0.3000 | 0.3846 | 0.5000 | 0.4348 | 5332 |
| fusion_attention | attention | gt | 0.2261 | 0.3929 | 0.1818 | 0.6000 | 0.2791 | 5887 |
| fusion_attention (test w/ cmSalGAN maps) | attention | gt→precomputed | 0.2261 | 0.3093 | 0.1471 | 0.5000 | 0.2273 | 5751 |
