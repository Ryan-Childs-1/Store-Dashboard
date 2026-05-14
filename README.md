# Store AI Dashboard

A GitHub-ready Streamlit dashboard for exploring store locations, state-level market statistics, national rollups, and AI-style forecasts using bundled population and GDP data plus an uploaded `Store List v1.csv` file.

## Files included in this repository

```text
store_ai_dashboard/
├── app.py
├── requirements.txt
├── README.md
└── 
    ├── US States Ranked by Population 2024.csv
    └── SAGDP2N__ALL_AREAS_1997_2020.csv
```

The app intentionally does **not** bundle `Store List v1.csv`. Upload it in the sidebar after launching the dashboard.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Upload this folder to a GitHub repository.
2. In Streamlit Cloud, select `app.py` as the entry point.
3. Add no secrets. The app runs locally from bundled CSVs and the uploaded store list.
4. Launch the app and upload `Store List v1.csv` in the sidebar.

## Notes

- The dashboard uses built-in state centroids with deterministic jitter for store map points because the store list contains addresses but no latitude/longitude columns.
- If future store files include `lat/lon`, `latitude/longitude`, or similar columns, the app will automatically use those exact coordinates.
- The forecasting engine is implemented from scratch as a lightweight LSTM-reservoir model. It uses explicit LSTM gate equations with fixed recurrent weights and a trained ridge readout, avoiding TensorFlow/PyTorch installation issues on Streamlit Cloud.


## Flat GitHub File Layout

This version keeps the bundled CSV files in the same folder as `app.py` for simple GitHub / Streamlit Cloud deployment. Do not place the population or GDP CSVs inside a `data` subfolder.
