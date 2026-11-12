# Placement Prediction Dashboard

An interactive machine-learning dashboard for exploring student placement data, training classification and regression models, and reviewing model performance.

## Overview

The application is built with Python, Dash, pandas, NumPy, Plotly, and scikit-learn. It includes a bundled placement dataset and a custom dark dashboard theme.

## Features

- Explore the bundled placement dataset and inspect its columns and distributions.
- Prepare data for machine-learning workflows.
- Train supported classification and regression estimators.
- Review model metrics and visual diagnostics.
- Explore dimensionality-reduction and anomaly-detection workflows where supported by the application.
- Interactive charts and tables powered by Plotly and Dash.

> Model scores depend on the selected target, features, split, and dataset. Do not quote a performance number unless it is reproduced from a documented run.

## Tech stack

- Python 3.10+
- Dash
- pandas, NumPy, SciPy
- scikit-learn
- Plotly
- openpyxl

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python dashboard.py
```

Open http://127.0.0.1:8050 in your browser.

## Dataset

The repository includes `placement_predict_50k_Dataset.xlsx`. Keep the original file unchanged when comparing experiments. For a different dataset, use the application's supported upload workflow.

## Project structure

```text
.
├── assets/
│   └── style.css
├── dashboard.py
├── placement_predict_50k_Dataset.xlsx
├── requirements.txt
└── README.md
```

## Reproducibility and responsible use

- Record the target column, selected features, random seed, and train/test split when comparing models.
- Avoid data leakage: fit preprocessing only on training data.
- Review class imbalance and per-class metrics instead of relying on accuracy alone.
- Treat predictions as decision support, not as guarantees of employment.
- Do not upload personally identifiable student data to an untrusted deployment.

## Current limitations

This is a local dashboard prototype. Before public deployment, add authentication and authorization, request-size limits, server-side session isolation, secure upload validation, observability, and a production deployment configuration. Review the cache/session implementation before hosting with multiple users.

## License

No license is currently specified. Contact the repository owner before reusing or redistributing the code or dataset.