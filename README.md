# Beehive Nectar Flow Prediction: Benchmarking Causal CNN, LSTM, and Causal Transformer under Inductive Bias and Iso-Parameter Constraints

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: 5--Seed Tested](https://img.shields.io/badge/Reproducibility-5--Seed%20Verified-brightgreen.svg)]()
[![Status: Complete](https://img.shields.io/badge/Status-Completed-success.svg)]()

---

## 📌 Executive Summary & Abstract

Predicting colony weight changes ($\Delta \text{weight}_{t+24}$, indicative of honeybee nectar flow and hive biomass dynamics) is a foundational task in precision apiculture and ecological time-series forecasting. This repository provides an end-to-end, rigorous empirical benchmark comparing three distinct deep temporal architectures under strict **iso-parameter (~48k–50k parameters)**, **identical hyperparameter tuning budgets (6 trials on validation)**, and **multi-seed statistical guarantees (5 independent seeds, 95% Student-$t$ confidence intervals)**:

1. **Causal Dilated Residual Convolutional Neural Network (TCN):** Designed with an analytical Receptive Field of $RF = 253 \ge 168$ steps (capturing both 24h circadian cycles and 7-day weather patterns).
2. **Long Short-Term Memory Network (LSTM):** 2-layer recurrent state-space model with Markovian hidden state compression.
3. **Causal Masked Transformer:** 3-layer self-attention network with dynamic pairwise cross-time attention and sinusoidal positional embeddings.

### 🔬 Key Scientific Findings

* **CNN and LSTM are Statistically Indistinguishable:** On the locked test set (Spring 2019), LSTM achieved a Test MAE of **$0.2292 \pm 0.0222\text{ kg/24h}$** and Causal CNN achieved **$0.2331 \pm 0.0196\text{ kg/24h}$** (Welch's $t$-test $p = 0.7257$, overlapping confidence intervals). The pre-training hypothesis that CNN would strictly outperform LSTM was **not supported**.
* **Transformer Fails under Small-Data & Long-Context Regimes:** Causal Transformer yielded a substantially higher Test MAE of **$0.3268 \pm 0.0250\text{ kg/24h}$** ($p < 0.0001$ against both CNN and LSTM). Despite achieving the lowest validation error during tuning ($0.0824$), its weak inductive bias led to severe overfitting and generalization collapse across seasonal shifts.
* **Context Window Ablation:** Expanding the temporal context from 1 day ($24\text{h}$) to 7 days ($168\text{h}$) provided **no significant gain** for CNN or LSTM, whereas Transformer suffered acute degradation ($0.239 \to 0.327\text{ kg/24h}$).
* **Gradient Vanishing in LSTM:** Backpropagation-Through-Time (BPTT) gradient norms decayed by nine orders of magnitude ($\text{lag}_{167} / \text{lag}_0 \approx 6.7 \times 10^{-10}$), yet this did not penalize accuracy because 24h diurnal dynamics dominate predictive power.
* **Few-Shot Domain Adaptation (Bad Schwartau):** Across $n \in \{2, 7, 30\}$ days of adaptation, no transfer strategy meaningfully outperformed naive baselines ($0.155–0.160\text{ kg}$), indicating that small target sample sizes ($N \le 59$) require stronger domain regularization.

---

## 📊 Benchmark Overview & Main Results

### 1. Main Model Comparison (Locked Test Set: Feb 24 – May 20, 2019)

| Architecture | Parameters | MACs ($T=168$) | Mean Test MAE (kg/24h) $\pm$ 95% CI | Test RMSE (kg/24h) | CPU Latency (p50 / p95) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Causal Dilated CNN** | **48,665** | 7.89 M | $0.2331 \pm 0.0196$ | $0.4199 \pm 0.0209$ | 1.10 ms / 1.64 ms |
| **LSTM (2-Layer)** | **48,305** | **7.62 M** | $\mathbf{0.2292 \pm 0.0222}$ | $\mathbf{0.4077 \pm 0.0207}$ | **0.58 ms / 0.90 ms** |
| **Causal Transformer** | **49,829** | 15.33 M | $0.3268 \pm 0.0250$ | $0.5538 \pm 0.0391$ | 2.04 ms / 2.76 ms |
| *Baseline: Training Mean* | — | — | 0.2622 | 0.4552 | — |
| *Baseline: Persistence ($\Delta w_{t-24}$)* | — | — | 0.2617 | 0.5038 | — |
| *Baseline: Zero Predictor* | — | — | 0.2645 | — | — |

<p align="center">
  <img src="results/fig_main.png" width="700" alt="Main Model Comparison">
  <br>
  <em>Figure 1: Locked Test Set evaluation across 5 independent seeds with Student-t 95% confidence intervals and baseline thresholds.</em>
</p>

### 2. Pairwise Hypothesis Testing (Welch's Two-Sample $t$-Test)

| Comparison | $\Delta\text{MAE}$ (Model 1 − Model 2) | Welch $t$-statistic | $p$-value | 95% CI Overlap? | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Causal CNN vs. LSTM** | $+0.0039\text{ kg}$ | $t = 0.363$ | **$p = 0.7257$** | **Yes** | No statistically significant difference |
| **Causal CNN vs. Transformer** | $-0.0937\text{ kg}$ | $t = -8.102$ | **$p = 0.0001$** | **No** | CNN significantly outperforms Transformer |
| **LSTM vs. Transformer** | $-0.0976\text{ kg}$ | $t = -7.940$ | **$p < 0.0001$** | **No** | LSTM significantly outperforms Transformer |

---

## 🏗️ Architecture Design & Receptive Field Derivation

### 1. Receptive Field Derivation
To capture both daily foraging cycles ($24\text{h}$) and weekly synoptic weather patterns ($7 \times 24 = 168\text{h}$), the architectural receptive field must strictly satisfy $RF \ge 168$. 

For an $L$-layer residual network with two dilated convolutions per block (kernel size $k=3$) and exponential dilations $d_i \in \{1, 2, 4, 8, 16, 32\}$:

$$RF = 1 + 2 \cdot (k - 1) \cdot \sum_{i=1}^{L} d_i = 1 + 2 \cdot (3 - 1) \cdot 63 = \mathbf{253} \ge 168$$

*(Verified empirically via input gradient propagation: exact non-zero gradient span $= 253$ steps).*

### 2. Analytical Parameter & Compute Complexity (Manual Derivations)

All formulas include biases and are verified against PyTorch forward tensor hooks (`src/models.py:count_macs`):

* **Causal CNN ($C=36$, 6 residual blocks, $k=3$):**
  $$\text{Params} = \underbrace{(9 \cdot 36 + 36)}_{\text{Input 1x1}} + 6 \cdot \underbrace{2 \cdot (36 \cdot 36 \cdot 3 + 36)}_{\text{Residual Block}} + \underbrace{(36 \cdot 32 + 32) + 33}_{\text{Regression Head}} = \mathbf{48,665}$$
  $$\text{MACs}(T=168) = 168 \cdot \left[ 9 \cdot 36 + 12 \cdot (36 \cdot 36 \cdot 3) \right] + (36 \cdot 32 + 32) = \mathbf{7,893,824}$$

* **LSTM ($H=60$, 2 stacked layers):**
  $$\text{Params} = 4 \cdot [60 \cdot (9 + 60) + 120] + 4 \cdot [60 \cdot (60 + 60) + 120] + (60 \cdot 32 + 32) + 33 = \mathbf{48,305}$$
  $$\text{MACs}(T=168) = 168 \cdot 4 \cdot 60 \cdot [(9 + 60) + (60 + 60)] + (60 \cdot 32 + 32) = \mathbf{7,622,432}$$

* **Causal Transformer ($d=44, d_{ff}=88$, 3 layers, 4 heads):**
  $$\text{Params} = (9 \cdot 44 + 44) + 3 \cdot \left[ 4 \cdot (44^2 + 44) + 2 \cdot (44 \cdot 88) + 88 + 44 + 4 \cdot 44 \right] + (44 \cdot 32 + 32) + 33 = \mathbf{49,829}$$
  $$\text{MACs}(T=168) = 168 \cdot (9 \cdot 44) + 3 \cdot \left[ 4 \cdot 168 \cdot 44^2 + 2 \cdot 168^2 \cdot 44 + 2 \cdot 168 \cdot 44 \cdot 88 \right] + (44 \cdot 32 + 32) = \mathbf{15,325,056}$$

---

## 📈 Deep-Dive Analyses & Visualizations

### 1. Context Window Sensitivity ($W \in \{24\text{h}, 72\text{h}, 168\text{h}\}$)
Evaluating model sensitivity to historical context length reveals that local $24\text{h}$ features contain the predominant predictive signal for next-day nectar flow. Longer contexts introduce noise without commensurate representation capacity.

<p align="center">
  <img src="results/fig_window.png" width="650" alt="Context Window Ablation">
  <br>
  <em>Figure 2: Test MAE vs. context window size (1, 3, and 7 days) across all architectures.</em>
</p>

### 2. Backpropagation-Through-Time Gradient Decay
Computing the mean input gradient norm $\|\partial \hat{y} / \partial x_t\|$ as a function of time lag ($t \to t-167$) demonstrates severe exponential vanishing in LSTM, contrasting with stable dilation pathways in CNN and persistent attention weights in Transformer.

| Model | Lag 0 (Now) | Lag 6 (-6h) | Lag 24 (-24h) | Lag 72 (-3d) | Lag 167 (-7d) | Decay Ratio ($\text{Lag}_{167} / \text{Lag}_0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Causal CNN** | $8.60 \times 10^{-2}$ | $5.50 \times 10^{-3}$ | $4.11 \times 10^{-3}$ | $7.61 \times 10^{-4}$ | $8.03 \times 10^{-6}$ | $9.34 \times 10^{-5}$ |
| **LSTM** | $4.86 \times 10^{-3}$ | $4.40 \times 10^{-3}$ | $1.76 \times 10^{-5}$ | $8.32 \times 10^{-9}$ | $3.24 \times 10^{-12}$ | $\mathbf{6.67 \times 10^{-10}}$ |
| **Causal Transformer** | $1.45 \times 10^{-3}$ | $3.57 \times 10^{-5}$ | $1.47 \times 10^{-4}$ | $1.01 \times 10^{-4}$ | $1.71 \times 10^{-3}$ | $1.18 \times 10^{0}$ |

<p align="center">
  <img src="results/fig_grad.png" width="850" alt="BPTT Gradient Decay">
  <br>
  <em>Figure 3: Temporal gradient backpropagation profile illustrating vanishing gradients across 168 hours.</em>
</p>

### 3. Few-Shot Domain Adaptation (Bad Schwartau Hive)
Transferring models pre-trained on Würzburg ($12\text{h}$ resampled grid) to the unobserved Bad Schwartau hive under limited supervision ($n \in \{2, 7, 30\}$ days):

<p align="center">
  <img src="results/fig_transfer.png" width="900" alt="Few-Shot Transfer Learning">
  <br>
  <em>Figure 4: Adaptation performance comparison (Scratch vs. Linear Probe vs. Full Fine-Tuning) against naive baselines.</em>
</p>

---

## 🗂️ Repository Structure

```
├── data/
│   ├── beehive_metrics.csv       # Kaggle Beehive Metrics sensor dataset (15-min raw)
│   ├── weather_wurzburg.csv      # Open-Meteo hourly meteorology (Würzburg, 2017–2019)
│   └── weather_schwartau.csv     # Open-Meteo hourly meteorology (Bad Schwartau, 2017)
├── src/
│   ├── data.py                   # Preprocessing, spike filtering (|Δw| > 2kg), feature scaling
│   ├── models.py                 # PyTorch implementations of Causal CNN, LSTM, Transformer & MAC hooks
│   ├── train.py                  # Standardized 5-seed training loop with Huber loss & early stopping
│   ├── analysis.py               # Context ablation, gradient norm BPTT, latency, and transfer modules
│   ├── summarize.py              # Statistical aggregations, Welch tests, and automated figure generation
│   └── weather.py                # Open-Meteo API automated weather collector
├── report/
│   ├── Kovan_Rapor_Final.docx    # Complete final formatted academic submission report (.docx)
│   ├── sonuc_ve_rapor.md         # Full markdown academic technical report
│   ├── hipotez.md                # Formal pre-training inductive bias hypothesis document
│   └── tasarim_ve_hesaplamalar.md# Detailed mathematical derivations of RF, Params, and MACs
├── results/
│   ├── fig_main.png              # Publication Figure: Main 5-seed model comparison
│   ├── fig_window.png            # Publication Figure: Context window ablation
│   ├── fig_grad.png              # Publication Figure: BPTT gradient decay curve
│   ├── fig_transfer.png          # Publication Figure: Few-shot transfer adaptation curve
│   ├── main_*.json               # Raw empirical metrics for all 5 seeds per architecture
│   ├── ckpt_*.pt                 # PyTorch model checkpoints for all trained seeds
│   └── tables.json               # Aggregated LaTeX/Markdown tables
├── run_pipeline_stages.py        # Automated stage orchestrator for reproduction
├── md2docx.py                    # Script compiling Markdown reports into formatted Word documents
├── status.py                     # Live pipeline monitoring CLI dashboard
└── README.md                     # Project documentation
```

---

## 🚀 Reproduction & Quickstart Guide

### 1. Environment Setup
```bash
git clone https://github.com/sarkannnn/beehive-nectar-flow-prediction.git
cd beehive-nectar-flow-prediction
pip install torch numpy scipy matplotlib python-docx requests
```

### 2. Verify Iso-Parameter Calibration & Manual MACs
```bash
python src/models.py
```
*Expected output: CNN (48,665), LSTM (48,305), Transformer (49,829) params; MACs match manual derivations.*

### 3. Run End-to-End Training & Analysis Pipeline
```bash
# Execute full benchmark (tuning + 5 seeds + ablations + figures + report):
python run_pipeline_stages.py
```

### 4. Inspect Live Status at Any Time
```bash
python status.py
```

---

## 📝 Citation & Academic Reference

If you utilize this benchmark, dataset preprocessing pipeline, or architectural implementations, please cite as follows:

```bibtex
@misc{mammadov2026beehive,
  author       = {Sarkan Mammadov},
  title        = {Beehive Nectar Flow Prediction: Benchmarking Causal CNN, LSTM, and Causal Transformer under Inductive Bias and Iso-Parameter Constraints},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/sarkannnn/beehive-nectar-flow-prediction}}
}
```

---
**License:** Distributed under the [MIT License](LICENSE).
