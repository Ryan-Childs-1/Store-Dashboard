# Store Performance & Market Dashboard

A GitHub-ready Streamlit dashboard for analyzing a retail store list against bundled state population and GDP data.

## Files in this folder

All files are intentionally kept in the same folder for GitHub / Streamlit Cloud compatibility:

```text
app.py
requirements.txt
README.md
US States Ranked by Population 2024.csv
SAGDP2N__ALL_AREAS_1997_2020.csv
.streamlit/config.toml
```

The app asks the user to upload the third file at runtime:

```text
Store List v1.csv
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `Store List v1.csv` in the sidebar.

## Main dashboard features

The previous AI forecast tab has been removed. The dashboard now focuses on explainable business metrics, benchmarking, and one advanced but transparent growth projection model:

1. Revenue per square foot
2. Year-over-year growth
3. Store density by population
4. Revenue per capita
5. Store productivity index
6. Volume stability
7. Category / market flag performance
8. Market opportunity score
9. Advanced growth projection model
10. Store age / maturity analysis

## Notes

- Store volume fields such as `24 Volume` are automatically scaled to dollars when the app detects that they are stored in thousands.
- If exact latitude/longitude columns are not provided in the store list, store locations are plotted using state centroids plus deterministic jitter so that locations remain visible on the map.
- The map supports state-level drilldown through the sidebar. Newer Streamlit versions also support point selection on the map markers.


## Advanced Growth Projection tab

The Growth Projection tab replaces the old Volume Bands view. It uses an explainable, deployment-safe model rather than a black-box neural forecast.

The model:

1. Calculates actual 2021-2024 store and state volume trends.
2. Builds state macro features from the bundled GDP and population files.
3. Uses a NumPy-based ridge regression model to estimate how macro conditions relate to recent store growth.
4. Blends the macro model with actual state/store momentum and national mean reversion.
5. Produces state-level and store-level projections with low/base/high uncertainty bands.

This keeps the model useful on GitHub and Streamlit Cloud without requiring TensorFlow, PyTorch, or scikit-learn.

## Latest interaction upgrades

- Removed the table/chart row-limit slider entirely. Tables and charts now include every row that passes the active filters.
- Added global filters in the sidebar that apply to the map, drilldown, side metrics, and all 10 tabs.
- Added always-visible sidebar metrics for the current filtered/drilldown view.
- Added useful filter controls for region, size band, volume band, red/blue, pick day, maturity, productivity band, numeric ranges, category/market flags, and search.
