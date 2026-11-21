# FedSVLM: Federated Learning with Fine-Tuned Small Vision-Language Models for Multimodal Fake News Detection

Code, logs and results for the term paper of the same title (Natural Language Processing course, Quy Nhon University, 2026).

- **Author:** Thi-Han Cao (Cao Thị Hân), Department of Mathematics and Statistics, Quy Nhon University, caothihan.2005vta@gmail.com
- **Supervisor:** Quang-Hung Le (lequanghung@qnu.edu.vn)

## Overview

We train Light-MFND, a small vision-language model for multimodal fake news detection (CLIP ViT-B/32 adapted with LoRA, with cross-attention fusion; Nguyen, Phan & Le, 2026), with federated averaging (FedAvg). Only the LoRA, fusion and classifier parameters are exchanged (2.07 M parameters, 8.27 MB per update). We call this setup **FedSVLM**. We also study **FedSVLM-LF**, a late-fusion variant in which an image-only and a text-only model are trained with FedAvg and their probabilities are averaged (1.97 MB per round in total).

Experiments use two datasets:

| Dataset | Clients | In-domain test | Out-of-domain (OOD) test |
|---|---|---|---|
| MiRAGeNews | 5 clients, Dirichlet label skew (α = 100, 0.5, 0.1) | `test1_nyt_mj` | 4 sets: BBC/CNN captions × DALL-E/SDXL images |
| FineFake | 4 clients = 4 topics (Politics, Business, Health, Society) | held-out 20% of the same topics | Entertainment, Conflict |

All federated runs are **simulated on a single machine** (Google Colab, NVIDIA T4). The privacy benefit discussed in the paper is structural only (raw data stay on each client); no formal privacy guarantee is claimed.

## Repository structure

```
FedSVLM/
├── README.md
├── requirements.txt
├── data_prep/
│   └── 00_MiRAGeNews_preprocessing.ipynb   # MiRAGeNews → CLIP encoding → memory-mapped .npy
├── notebooks/
│   ├── 01_MiRAGeNews_FedSVLM.ipynb         # all MiRAGeNews experiments
│   └── 02_FineFake_FedSVLM.ipynb           # FineFake download, preprocessing and experiments
└── results/
    ├── results_miragenews.json             # 45 runs: metrics per test set (+ per-round logs)
    └── results_finefake.json               # all FineFake runs and analyses
```

Each notebook starts with a description of what it does, the run order, notes on how the saved outputs were produced, and a glossary for the Vietnamese labels that appear in code comments and printed outputs. All notebooks are saved **with their outputs**, so they can be read without re-running.

## Data

The datasets are not included in this repository (size and licensing). Download them from the official sources:

- **MiRAGeNews** (Huang et al., 2024): Hugging Face `anson-huang/mirage-news`, revision `b5a7e734850b4ec623ddee018a1d9e097fe248ef`. Preprocess with `data_prep/00_MiRAGeNews_preprocessing.ipynb`.
- **FineFake** (Zhou et al., 2024): <https://github.com/Accuser907/FineFake>. Notebook 2 downloads the official release and builds the topic-based split (cells F1a–F2).

## How to run

1. Open the notebooks in Google Colab and select a GPU runtime (data preparation cells can run on CPU).
2. The notebooks keep data, checkpoints and results in Google Drive under `MyDrive/NLP_project/fed/`. Change the `FED` path in the first cells if you use another location.
3. Run `data_prep/00_MiRAGeNews_preprocessing.ipynb`, then `notebooks/01_MiRAGeNews_FedSVLM.ipynb` (it creates `init.pt`, the shared LoRA initialisation also used by notebook 2), then `notebooks/02_FineFake_FedSVLM.ipynb`.
4. Every experiment cell writes its own checkpoint and appends its result to the JSON file, so cells can be run one at a time and resumed after a disconnected session.

Settings shared by all runs: AdamW, learning rate 1e-5, weight decay 0.01, batch size 16, fp16; LoRA r = 8, α = 16, dropout 0.1 on `q_proj` and `v_proj`; FedAvg with 20 rounds × 1 local epoch; the best round is chosen by validation AUC.

## Main results

AUC in %. "± " is the standard deviation over runs (seeds); single runs have no ±.

### MiRAGeNews

| Method | Runs | In-domain AUC | OOD AUC | MB per update |
|---|---|---|---|---|
| Centralized Light-MFND | 3 | 100.0 | 80.6 ± 1.1 | – |
| Local-only (α = 0.5, mean of 5 clients) | 1 | 89.0 | 66.4 | – |
| FedSVLM, α = 100 | 3 | 100.0 | 77.9 ± 0.9 | 8.27 |
| FedSVLM, α = 0.5, split 0 | 3 | 100.0 | 86.1 ± 0.4 | 8.27 |
| FedSVLM, α = 0.5, split 1 / split 2 | 1 / 1 | 100.0 / 100.0 | 81.2 / 81.2 | 8.27 |
| FedSVLM, α = 0.1 | 3 | 100.0 | 83.6 ± 0.7 | 8.27 |
| FedAvg, full fine-tuning, α = 0.5 | 1 | 99.7 | 72.4 | 611.4 |
| Centralized late fusion (image + text) | 2 | 100.0 | 91.2 ± 0.4 | – |
| FedSVLM-LF, α = 0.5, split 0 | 1 | 99.9 | 86.9 | 1.97 |
| FedSVLM-LF, α = 0.5, split 2 | 3 | 100.0 | 92.0 ± 0.2 | 1.97 |
| TF-IDF + SVM (text only) | 1 | 92.8 | 87.0 | – |

Caveats: OOD results depend strongly on the client partition (the high score of split 0 does not hold on splits 1 and 2). Test sets test2/test4 and test3/test5 share most captions and real images (see cell 41a), so the four OOD scores are not independent.

### FineFake

| Method | Test AUC | OOD AUC |
|---|---|---|
| Centralized Light-MFND (20 epochs) | 80.8 | 77.1 |
| Local-only (mean of 4 topic clients) | 70.6 | 67.5 |
| FedSVLM | 79.3 | 76.1 |
| FedSVLM-LF | 82.3 | 79.8 |
| TF-IDF + SVM (text only) | 80.3 | 77.7 |
| News source only (no text, no image) | 74.0 | 73.9 |

A classifier that sees only the news source already reaches 74 AUC, and all models reach only about 59 AUC on average **within** each source. Source-balanced client resampling raises the mean within-source AUC of FedSVLM-LF from 59.0 to 64.1 but lowers pooled AUC from 82.4 to 74.8 (two runs each). Adding the knowledge embeddings provided with FineFake did not improve any metric.

## Acknowledgements

The model architecture follows the Light-MFND repository (Nguyen, Phan & Le, 2026). We thank the authors of MiRAGeNews and FineFake for releasing their data.
