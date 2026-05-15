# Store Performance & Market Dashboard

A GitHub-ready Streamlit dashboard for analyzing `Store List v1.csv` against bundled state population and BEA state GDP data.

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

## Major upgrade in this version

This version turns the app into a strategic store-performance system. It adds an explicit **Major Insights** tab that summarizes the uploaded store list in the same style as an analyst readout, then builds supporting tabs for the key questions behind those insights.

The app now emphasizes:

- store productivity and revenue per square foot
- actual 2021-2024 revenue trends
- state risk and decline monitoring
- store format efficiency
- new-store vs mature-store analysis
- state saturation and white-space scoring
- GDP-based outdoor market fit
- market penetration against retail and outdoor-relevant GDP
- productivity scoring
- flag / merchandising performance
- store archetype clustering
- best-practice stores worth studying
- underperformer diagnostics
- macro-adjusted growth projection
- stability analysis

## GDP segment logic

Because the stores sell hunting, camping, outdoor gear, and firearms, total GDP alone is too broad. The app creates an **Outdoor Retail Macro Score** using the most relevant available GDP categories in the BEA file:

| GDP segment | Weight | Purpose |
|---|---:|---|
| Retail trade | 30% | Best direct proxy for the state retail economy |
| Accommodation and food services | 20% | Tourism, travel, camping-trip, and destination-market proxy |
| Arts, entertainment, and recreation | 15% | Closest GDP category to recreation activity |
| Agriculture, forestry, fishing and hunting | 15% | Rural, hunting, fishing, forestry, and outdoor-culture proxy |
| Natural resources and mining | 10% | Rugged/resource-state lifestyle and rural economy proxy |
| All industry total | 10% | General economic strength control |

The score is then used in market-fit, white-space, best-practice, penetration, and growth-projection views.

## Dashboard tabs

1. Major Insights
2. Revenue / Sq. Ft.
3. Growth & Risk
4. Store Format Efficiency
5. Maturity Cohorts
6. Saturation & White Space
7. Outdoor Market Fit / GDP
8. Market Penetration
9. Productivity Index
10. Flag Performance
11. Store Archetypes
12. Best Practices
13. Underperformer Diagnostic
14. Growth Projection
15. Stability

## Notes

- Store volume fields such as `24 Volume` are automatically scaled to dollars when the app detects that they are stored in thousands.
- If exact latitude/longitude columns are not provided in the store list, store locations are plotted using state centroids plus deterministic jitter so that locations remain visible on the map.
- Tables and charts are uncapped: every row passing the active filters is included.
- Global sidebar filters apply to the map, state drilldown, side metrics, and every tab.
- The Growth Projection tab uses an explainable NumPy-based ridge/blended model rather than TensorFlow, PyTorch, or scikit-learn so it remains easy to deploy on Streamlit Cloud.
