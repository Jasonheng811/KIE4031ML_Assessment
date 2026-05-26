# KIE4031 — Stock Price Prediction with RNN (LSTM vs GRU)

Final Summative Assessment, Universiti Malaya, Semester II 2025/2026.

## Overview

This project predicts **Apple Inc. (AAPL)** daily closing prices using a Long Short-Term Memory (LSTM) recurrent neural network, then critically compares it against a Gated Recurrent Unit (GRU) alternative.

The assessment is structured around four rubric questions worth **30 marks**:

| Q | Task | Marks |
|---|------|-------|
| 1 | Data collection & preprocessing | 10 |
| 2 | Investigation of RNN technique characteristics | 5 |
| 3 | Model development, evaluation & visualisations | 5 |
| 4 | Critical analysis & alternative model comparison | 10 |

## Repository contents

```
KIE4031_Assessment/
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── notebook.ipynb        # Main deliverable — all code, plots, inline narrative
├── report.md             # Standalone written report
├── data/AAPL.csv         # Cached dataset (downloaded by notebook on first run)
└── figures/              # Exported plots used by report.md
```

## How to run

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook notebook.ipynb

# 3. Run all cells: Kernel > Restart & Run All
```

The notebook downloads ~10 years of AAPL OHLCV data on first run, caches it
to `data/AAPL.csv`, trains both LSTM and GRU models, runs the hyperparameter
sweep + multi-seed robustness + log-returns experiment + classical
baselines, and saves all figures to `figures/`. Full execution takes
~8–12 minutes on CPU.

## Producing PDFs for submission

`report.html` and `notebook.html` are already rendered. To produce PDFs:

1. Open `report.html` (or `notebook.html`) in any browser.
2. Press **Ctrl+P** -> destination **"Save as PDF"** -> Save.

This is the most reliable cross-platform way to produce print-quality PDFs
without needing `pandoc`, LaTeX or other heavy toolchains.

## Regenerating the HTML files

```powershell
# Rebuild HTML view of the report
python build_html.py

# Rebuild HTML view of the notebook (includes all plots/outputs)
jupyter nbconvert --to html notebook.ipynb

# Optional: rebuild notebook.ipynb itself from build_notebook.py and re-execute
python build_notebook.py
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace
```

## Stack

- **Python 3.11**
- **TensorFlow / Keras** — model definition & training
- **yfinance** — dataset
- **pandas, numpy, scikit-learn** — preprocessing
- **matplotlib** — visualisation
