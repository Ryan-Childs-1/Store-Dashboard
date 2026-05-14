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

The previous AI forecast tab has been removed. The dashboard now focuses on explainable business metrics and benchmarking:

1. Revenue per square foot
2. Year-over-year growth
3. Store density by population
4. Revenue per capita
5. Store productivity index
6. Volume stability
7. Category / market flag performance
8. Market opportunity score
9. Volume band movement
10. Store age / maturity analysis

## Notes

- Store volume fields such as `24 Volume` are automatically scaled to dollars when the app detects that they are stored in thousands.
- If exact latitude/longitude columns are not provided in the store list, store locations are plotted using state centroids plus deterministic jitter so that locations remain visible on the map.
- The map supports state-level drilldown through the sidebar. Newer Streamlit versions also support point selection on the map markers.
