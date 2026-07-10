# BCI Augmentation Explorer

This project implements a lightweight, fully interactive EEG augmentation demo with a precomputed data pipeline and a Streamlit dashboard.

## Structure

- `scripts/build_demo_data.py` builds a pickled demo payload from synthetic or source EEG data.
- `app/streamlit_app.py` renders the interactive dashboard.
- `data/raw/` holds the raw inputs if available; the builder will synthesize placeholder files when they are missing.

## Setup

```bash
cd /Users/shriram/Desktop/EEG-GAN\ demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_demo_data.py
streamlit run app/streamlit_app.py
```

## Notes

- The demo intentionally avoids live training and instead uses precomputed metrics and simulation curves.
- The generated artifact is stored at `data/demo_data.pkl`.
