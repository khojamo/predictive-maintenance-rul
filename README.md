# Predictive Maintenance RUL

This project is a predictive maintenance demo that scores failure risk and remaining useful life (RUL) from CMAPSS-style sensor time-series data. It includes a FastAPI backend, a Streamlit UI, and prebuilt ML artifacts for inference.

## Quick start (auto-install)

Requirements: Python 3.11+

Start the API:

```bash
python run.py api
```

Start the UI (in another terminal):

```bash
python run.py ui
```

`run.py` asks whether to create a local `.venv` or use your current environment, then installs any missing dependencies before launching.

## Workflow (custom data)

1) Open the UI and upload your dataset (CSV or Excel).  
2) Map your columns to the canonical feature names.  
3) Train and evaluate. Artifacts are saved to `models/` and `reports/`.  
4) Score new data and run drift checks from the UI.

## Synthetic data generator

Generate a large synthetic dataset (CSV, Excel, or both):

```bash
python -m predictive_maintenance.scripts_generate_synthetic --format both --n-units 2000 --cycles 1000
```

Excel output is automatically split into multiple files if it exceeds Excel's row limit.

## Notes

- The UI calls the API at `http://127.0.0.1:8000` by default.
- You can pass extra arguments through `run.py` after `--`, for example:

```bash
python run.py api -- --host 0.0.0.0 --port 8000
```
