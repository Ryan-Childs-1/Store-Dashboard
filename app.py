# app.py
# ============================================================
# Store Performance & Market Dashboard
# GitHub/Streamlit-ready dashboard using:
#   1) bundled US population data
#   2) bundled BEA GDP-by-state data
#   3) user-uploaded Store List v1.csv
#
# All files are expected in the same folder as app.py.
# ============================================================

from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# App constants
# -----------------------------
APP_NAME = "Store Performance & Market Dashboard"
BASE_DIR = Path(__file__).resolve().parent
POP_FILE = BASE_DIR / "US States Ranked by Population 2024.csv"
GDP_FILE = BASE_DIR / "SAGDP2N__ALL_AREAS_1997_2020.csv"

STATE_ABBR: Dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}
ABBR_STATE = {v: k for k, v in STATE_ABBR.items()}

STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419), "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123), "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141), "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074), "HI": (21.094318, -157.498337), "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278), "IA": (42.011539, -93.210526),
    "KS": (38.526600, -96.726486), "KY": (37.668140, -84.670067), "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101), "MA": (42.230171, -71.530106),
    "MI": (43.326618, -84.536095), "MN": (45.694454, -93.900192), "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353), "NE": (41.125370, -98.268082),
    "NV": (38.313515, -117.055374), "NH": (43.452492, -71.563896), "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051), "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012), "OH": (40.388783, -82.764915), "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755), "RI": (41.680893, -71.511780),
    "SC": (33.856892, -80.945007), "SD": (44.299782, -99.438828), "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434), "VT": (44.045876, -72.710686),
    "VA": (37.769337, -78.169968), "WA": (47.400902, -121.490494), "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490), "DC": (38.9072, -77.0369),
}

FLAG_COLUMNS = [
    "Ocean", "Optics Move", "Year Round Kayaks", "Disaster Depots", "Analyst", "New", "Denali",
    "Power Alley", "Metro", "Ammo Hub", "Turkey", "Bear", "Elk", "Moose", "Dove", "Trapping",
    "Predator", "Goose", "Waterfowl",
]
VOLUME_COLUMNS = ["21 Volume", "22 Volume", "23 Volume", "24 Volume"]
VOLUME_YEAR_MAP = {"21 Volume": 2021, "22 Volume": 2022, "23 Volume": 2023, "24 Volume": 2024}

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🗺️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 3.25rem; padding-bottom: 2rem;}
    .small-note {font-size: .86rem; color: #777;}
    .big-title {font-size: 2.05rem; font-weight: 800; line-height: 1.25; padding-top: .75rem; margin-top: .25rem; margin-bottom: .35rem; overflow: visible;}
    .section-title {font-size: 1.22rem; font-weight: 750; margin-top: .8rem; margin-bottom: .4rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 16px; padding: .75rem .9rem; background: rgba(128,128,128,.04);}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# General utilities
# -----------------------------
def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", s)


def clean_state_name(x: object) -> str:
    return clean_text(x)


def safe_num(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip(), errors="coerce")


def yes_like(x: object) -> bool:
    s = clean_text(x).lower()
    return s in {"yes", "done", "y", "true", "1", "x", "new"}


def fmt_money(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000:
        return f"{sign}${x/1_000_000_000:,.2f}B"
    if x >= 1_000_000:
        return f"{sign}${x/1_000_000:,.2f}M"
    if x >= 1_000:
        return f"{sign}${x/1_000:,.1f}K"
    return f"{sign}${x:,.0f}"


def fmt_num(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.0f}"


def fmt_float(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.{digits}f}"


def fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x)*100:,.1f}%"


def rank_pct(s: pd.Series, higher_better: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(np.where(s.notna(), 50.0, np.nan), index=s.index)
    return s.rank(pct=True, ascending=not higher_better) * 100


def deterministic_jitter(n: int, seed_key: str, scale_lat: float = 1.25, scale_lon: float = 1.65) -> np.ndarray:
    seed = abs(hash(seed_key)) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, max(n, 1), endpoint=False)
    rng.shuffle(angles)
    radii = np.sqrt(rng.uniform(0.05, 1.0, max(n, 1)))
    return np.c_[np.sin(angles) * radii * scale_lat, np.cos(angles) * radii * scale_lon][:n]


def parse_open_date(v: object) -> pd.Timestamp:
    if pd.isna(v) or str(v).strip() == "":
        return pd.NaT
    # Store file appears to use Excel serials. Try that first when numeric.
    try:
        f = float(v)
        if 20_000 <= f <= 60_000:
            return pd.to_datetime(f, unit="D", origin="1899-12-30")
    except Exception:
        pass
    return pd.to_datetime(v, errors="coerce")


def parse_band_midpoint(band: object) -> Tuple[float, float, float]:
    """Return lower, upper, midpoint in dollars from bands like 12M-15M or Less than 6M."""
    s = clean_text(band).lower().replace("$", "")
    if not s or s == "-":
        return np.nan, np.nan, np.nan
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    mult = 1_000_000 if "m" in s else 1.0
    if "less" in s and nums:
        lo, hi = 0.0, nums[0] * mult
    elif ("+" in s or "greater" in s or "more" in s) and nums:
        lo, hi = nums[0] * mult, np.nan
    elif len(nums) >= 2:
        lo, hi = nums[0] * mult, nums[1] * mult
    elif len(nums) == 1:
        lo, hi = nums[0] * mult, np.nan
    else:
        return np.nan, np.nan, np.nan
    mid = (lo + hi) / 2 if not pd.isna(hi) else lo
    return lo, hi, mid


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_population() -> pd.DataFrame:
    if not POP_FILE.exists():
        return pd.DataFrame()
    pop = pd.read_csv(POP_FILE)
    pop.columns = [clean_text(c) for c in pop.columns]
    if "US State" in pop.columns:
        pop["State"] = pop["US State"].map(clean_state_name)
    else:
        pop["State"] = pop.iloc[:, 0].map(clean_state_name)
    pop["State Abbr"] = pop["State"].map(STATE_ABBR)
    for c in ["Population 2024", "Population 2023", "Growth Rate", "% of US", "Density (/mile2)"]:
        if c in pop.columns:
            pop[c] = safe_num(pop[c])
    return pop


@st.cache_data(show_spinner=False)
def load_gdp() -> pd.DataFrame:
    if not GDP_FILE.exists():
        return pd.DataFrame()
    gdp = pd.read_csv(GDP_FILE)
    gdp.columns = [clean_text(c) for c in gdp.columns]
    if "GeoName" not in gdp.columns:
        return pd.DataFrame()
    gdp["GeoName"] = gdp["GeoName"].astype(str).str.replace("*", "", regex=False).str.strip()
    gdp["Description"] = gdp.get("Description", "").astype(str).str.strip()
    gdp["State"] = gdp["GeoName"].map(clean_state_name)
    gdp["State Abbr"] = gdp["State"].map(STATE_ABBR)
    for c in [c for c in gdp.columns if re.fullmatch(r"\d{4}", str(c))]:
        gdp[c] = safe_num(gdp[c])
    return gdp


def read_uploaded_store_file(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"Could not read uploaded file. Last error: {last_err}")


def normalize_store_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_text(c) for c in df.columns]
    if "State" not in df.columns:
        raise ValueError("The uploaded store file needs a State column.")

    df["State"] = df["State"].map(clean_state_name)
    df["State Abbr"] = df["State"].map(STATE_ABBR)
    is_abbr = df["State Abbr"].isna() & df["State"].str.len().eq(2)
    df.loc[is_abbr, "State Abbr"] = df.loc[is_abbr, "State"].str.upper().where(df.loc[is_abbr, "State"].str.upper().isin(ABBR_STATE))
    df["State"] = df["State Abbr"].map(ABBR_STATE).fillna(df["State"])

    for c in ["Sq. Footage", "Grand Opening"] + VOLUME_COLUMNS:
        if c in df.columns:
            df[c] = safe_num(df[c])

    # Detect whether volumes are stored in thousands. Example: 12,544 aligns with a 12M-15M band.
    scale = 1.0
    if "24 Volume" in df.columns and "Volume Band" in df.columns:
        med = df["24 Volume"].dropna().median()
        if pd.notna(med) and med < 500_000 and df["Volume Band"].astype(str).str.contains("M", case=False, na=False).any():
            scale = 1_000.0
    df.attrs["volume_scale"] = scale
    for c in VOLUME_COLUMNS:
        if c in df.columns:
            df[c + " Dollars"] = df[c] * scale

    if "Sq. Footage" in df.columns and "24 Volume Dollars" in df.columns:
        df["Revenue per Sq. Ft."] = df["24 Volume Dollars"] / df["Sq. Footage"].replace(0, np.nan)

    for c in VOLUME_COLUMNS:
        cd = c + " Dollars"
        if cd not in df.columns and c in df.columns:
            df[cd] = df[c]

    if "21 Volume Dollars" in df.columns and "24 Volume Dollars" in df.columns:
        years = [c + " Dollars" for c in VOLUME_COLUMNS if c + " Dollars" in df.columns]
        df["Volume Avg 2021-2024"] = df[years].mean(axis=1)
        df["Volume Std 2021-2024"] = df[years].std(axis=1)
        df["Volume CV"] = df["Volume Std 2021-2024"] / df["Volume Avg 2021-2024"].replace(0, np.nan)
        df["Stability Score"] = (100 * (1 - df["Volume CV"].clip(lower=0, upper=1))).clip(lower=0, upper=100)
        if len(years) >= 2:
            df["YoY Growth 24 vs 23"] = (df["24 Volume Dollars"] - df["23 Volume Dollars"]) / df["23 Volume Dollars"].replace(0, np.nan)
            df["YoY Growth 23 vs 22"] = (df["23 Volume Dollars"] - df["22 Volume Dollars"]) / df["22 Volume Dollars"].replace(0, np.nan)
            df["YoY Growth 22 vs 21"] = (df["22 Volume Dollars"] - df["21 Volume Dollars"]) / df["21 Volume Dollars"].replace(0, np.nan)
            start = df["21 Volume Dollars"].replace(0, np.nan)
            df["CAGR 21-24"] = (df["24 Volume Dollars"] / start) ** (1/3) - 1

    if "Grand Opening" in df.columns:
        df["Grand Opening Date"] = df["Grand Opening"].map(parse_open_date)
        df["Grand Opening Year"] = df["Grand Opening Date"].dt.year
        df["Store Age"] = 2026 - df["Grand Opening Year"]
        df["Maturity Band"] = pd.cut(
            df["Store Age"],
            bins=[-1, 2, 5, 10, 20, 200],
            labels=["New: 0-2 yrs", "Ramp: 3-5 yrs", "Developed: 6-10 yrs", "Mature: 11-20 yrs", "Legacy: 21+ yrs"],
        )

    if "Volume Band" in df.columns:
        lows, highs, mids = zip(*df["Volume Band"].map(parse_band_midpoint))
        df["Band Lower"] = lows
        df["Band Upper"] = highs
        df["Band Midpoint"] = mids
        df["Distance to Next Band"] = df["Band Upper"] - df.get("24 Volume Dollars", np.nan)
        df["Distance Above Lower Band"] = df.get("24 Volume Dollars", np.nan) - df["Band Lower"]
        df["Band Position"] = (df.get("24 Volume Dollars", np.nan) - df["Band Lower"]) / (df["Band Upper"] - df["Band Lower"]).replace(0, np.nan)

    for c in FLAG_COLUMNS:
        if c in df.columns:
            df[c + " Flag"] = df[c].map(yes_like)

    # Store coordinates: exact if available; otherwise centroid+jitter by state for a useful map.
    lat_candidates = [c for c in df.columns if c.lower() in {"lat", "latitude", "store_latitude"}]
    lon_candidates = [c for c in df.columns if c.lower() in {"lon", "lng", "long", "longitude", "store_longitude"}]
    if lat_candidates and lon_candidates:
        df["Latitude"] = safe_num(df[lat_candidates[0]])
        df["Longitude"] = safe_num(df[lon_candidates[0]])
        df["Coordinate Source"] = "Uploaded coordinates"
    else:
        df["Latitude"] = np.nan
        df["Longitude"] = np.nan
        df["Coordinate Source"] = "State centroid + deterministic jitter"
        for abbr, idx in df.groupby("State Abbr", dropna=True).groups.items():
            if abbr in STATE_CENTROIDS:
                lat, lon = STATE_CENTROIDS[abbr]
                indices = list(idx)
                jit = deterministic_jitter(len(indices), str(abbr))
                df.loc[indices, "Latitude"] = lat + jit[:, 0]
                df.loc[indices, "Longitude"] = lon + jit[:, 1]
    return df


# -----------------------------
# Metric construction
# -----------------------------
def build_state_summary(stores: pd.DataFrame, pop: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
    if stores.empty:
        base = pd.DataFrame({"State Abbr": list(ABBR_STATE.keys()), "State": [ABBR_STATE[a] for a in ABBR_STATE]})
    else:
        id_col = "Store Number - Name" if "Store Number - Name" in stores.columns else "State"
        base = stores.groupby(["State Abbr", "State"], dropna=False).agg(
            Store_Count=(id_col, "count"),
            Total_Sq_Footage=("Sq. Footage", "sum") if "Sq. Footage" in stores.columns else (id_col, "count"),
            Avg_Sq_Footage=("Sq. Footage", "mean") if "Sq. Footage" in stores.columns else (id_col, "count"),
            Volume_2024=("24 Volume Dollars", "sum") if "24 Volume Dollars" in stores.columns else (id_col, "count"),
            Volume_2023=("23 Volume Dollars", "sum") if "23 Volume Dollars" in stores.columns else (id_col, "count"),
            Volume_2022=("22 Volume Dollars", "sum") if "22 Volume Dollars" in stores.columns else (id_col, "count"),
            Volume_2021=("21 Volume Dollars", "sum") if "21 Volume Dollars" in stores.columns else (id_col, "count"),
            Avg_Rev_per_SqFt=("Revenue per Sq. Ft.", "mean") if "Revenue per Sq. Ft." in stores.columns else (id_col, "count"),
            Avg_Stability=("Stability Score", "mean") if "Stability Score" in stores.columns else (id_col, "count"),
            Avg_Store_Age=("Store Age", "mean") if "Store Age" in stores.columns else (id_col, "count"),
        ).reset_index()

    if not pop.empty and "State Abbr" in pop.columns:
        pop_cols = [c for c in ["State Abbr", "Population 2024", "Population 2023", "Growth Rate", "Density (/mile2)"] if c in pop.columns]
        base = base.merge(pop[pop_cols].dropna(subset=["State Abbr"]).drop_duplicates("State Abbr"), on="State Abbr", how="outer")

    if not gdp.empty and "Description" in gdp.columns:
        gdp_total = gdp[gdp["Description"].eq("All industry total")].copy()
        if "2020" in gdp_total.columns:
            gdp_small = gdp_total[["State Abbr", "2020"]].dropna(subset=["State Abbr"]).rename(columns={"2020": "GDP_2020_Millions"})
            base = base.merge(gdp_small, on="State Abbr", how="left")

    base["State"] = base["State"].fillna(base["State Abbr"].map(ABBR_STATE))
    for c in ["Store_Count", "Total_Sq_Footage", "Avg_Sq_Footage", "Volume_2024", "Volume_2023", "Volume_2022", "Volume_2021"]:
        if c in base.columns:
            base[c] = base[c].fillna(0)
    base["Volume_per_Store"] = base["Volume_2024"] / base["Store_Count"].replace(0, np.nan)
    base["Revenue_per_SqFt"] = base["Volume_2024"] / base["Total_Sq_Footage"].replace(0, np.nan)
    base["Growth_24_vs_23"] = (base["Volume_2024"] - base["Volume_2023"]) / base["Volume_2023"].replace(0, np.nan)
    base["Growth_23_vs_22"] = (base["Volume_2023"] - base["Volume_2022"]) / base["Volume_2022"].replace(0, np.nan)
    base["Growth_22_vs_21"] = (base["Volume_2022"] - base["Volume_2021"]) / base["Volume_2021"].replace(0, np.nan)
    base["CAGR_21_24"] = (base["Volume_2024"] / base["Volume_2021"].replace(0, np.nan)) ** (1/3) - 1
    base["Stores_per_1M_People"] = base["Store_Count"] / (base["Population 2024"] / 1_000_000).replace(0, np.nan)
    base["Population_per_Store"] = base["Population 2024"] / base["Store_Count"].replace(0, np.nan)
    base["Revenue_per_Capita"] = base["Volume_2024"] / base["Population 2024"].replace(0, np.nan)
    base["Revenue_per_GDP_Million"] = base["Volume_2024"] / base["GDP_2020_Millions"].replace(0, np.nan)

    # State Opportunity Score: high population/GDP/growth, low store density, strong existing revenue per store.
    base["Opportunity_Score"] = (
        0.25 * rank_pct(base.get("Population 2024"), True) +
        0.20 * rank_pct(base.get("GDP_2020_Millions"), True) +
        0.20 * rank_pct(base.get("Stores_per_1M_People"), False) +
        0.20 * rank_pct(base.get("Volume_per_Store"), True) +
        0.15 * rank_pct(base.get("CAGR_21_24"), True)
    )
    base["Market_Status"] = pd.cut(
        base["Opportunity_Score"],
        bins=[-1, 40, 60, 75, 100],
        labels=["Low Priority", "Watch", "Expansion Candidate", "High Opportunity"],
    )
    return base.sort_values("Store_Count", ascending=False)


def add_store_productivity_scores(stores: pd.DataFrame) -> pd.DataFrame:
    df = stores.copy()
    df["Productivity Index"] = (
        0.40 * rank_pct(df.get("Revenue per Sq. Ft."), True) +
        0.25 * rank_pct(df.get("24 Volume Dollars"), True) +
        0.20 * rank_pct(df.get("CAGR 21-24"), True) +
        0.15 * rank_pct(df.get("Stability Score"), True)
    )
    df["Productivity Band"] = pd.cut(
        df["Productivity Index"], bins=[-1, 35, 55, 75, 100], labels=["Needs Review", "Average", "Strong", "Leader"]
    )
    return df


def scope_data(stores: pd.DataFrame, selected_state: str) -> pd.DataFrame:
    if selected_state == "ALL":
        return stores.copy()
    return stores[stores["State Abbr"].eq(selected_state)].copy()


def state_row(summary: pd.DataFrame, selected_state: str) -> pd.Series:
    if selected_state == "ALL":
        return pd.Series({
            "State": "United States",
            "State Abbr": "ALL",
            "Store_Count": summary["Store_Count"].sum(),
            "Total_Sq_Footage": summary["Total_Sq_Footage"].sum(),
            "Volume_2024": summary["Volume_2024"].sum(),
            "Volume_2023": summary["Volume_2023"].sum(),
            "Population 2024": summary["Population 2024"].sum(skipna=True),
            "GDP_2020_Millions": summary["GDP_2020_Millions"].sum(skipna=True),
            "Revenue_per_SqFt": summary["Volume_2024"].sum() / summary["Total_Sq_Footage"].sum() if summary["Total_Sq_Footage"].sum() else np.nan,
            "Growth_24_vs_23": (summary["Volume_2024"].sum() - summary["Volume_2023"].sum()) / summary["Volume_2023"].sum() if summary["Volume_2023"].sum() else np.nan,
            "Stores_per_1M_People": summary["Store_Count"].sum() / (summary["Population 2024"].sum(skipna=True) / 1_000_000) if summary["Population 2024"].sum(skipna=True) else np.nan,
            "Revenue_per_Capita": summary["Volume_2024"].sum() / summary["Population 2024"].sum(skipna=True) if summary["Population 2024"].sum(skipna=True) else np.nan,
        })
    hit = summary[summary["State Abbr"].eq(selected_state)]
    return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)


# -----------------------------
# Plot helpers
# -----------------------------
def metric_choropleth(summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.choropleth(
        summary,
        locations="State Abbr",
        locationmode="USA-states",
        color=metric,
        scope="usa",
        hover_name="State",
        hover_data={
            "State Abbr": True,
            "Store_Count": ":,.0f" if "Store_Count" in summary.columns else False,
            "Volume_2024": ":,.0f" if "Volume_2024" in summary.columns else False,
            metric: ":,.2f",
        },
        title=title,
    )
    # Add clickable centroid markers. Click/select one to drill into a state.
    map_points = summary.dropna(subset=["State Abbr"]).copy()
    map_points["lat"] = map_points["State Abbr"].map(lambda a: STATE_CENTROIDS.get(a, (np.nan, np.nan))[0])
    map_points["lon"] = map_points["State Abbr"].map(lambda a: STATE_CENTROIDS.get(a, (np.nan, np.nan))[1])
    map_points = map_points.dropna(subset=["lat", "lon"])
    fig.add_trace(go.Scattergeo(
        lat=map_points["lat"], lon=map_points["lon"], mode="markers+text",
        text=map_points["State Abbr"], textposition="middle center",
        marker=dict(size=np.clip(map_points["Store_Count"].fillna(0) * 3 + 9, 9, 30), color="rgba(20,20,20,.35)", line=dict(width=1, color="white")),
        customdata=map_points[["State Abbr"]].to_numpy(),
        hovertemplate="%{text}<br>Stores: %{marker.size}<extra>Click/select to drill in</extra>",
        name="Clickable states",
    ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def store_location_map(stores: pd.DataFrame, title: str) -> go.Figure:
    if stores.empty or "Latitude" not in stores.columns:
        fig = go.Figure()
        fig.update_layout(height=420, title="No stores available for this view")
        return fig
    hover_cols = [c for c in ["Store Number - Name", "City", "State", "Address", "24 Volume Dollars", "Revenue per Sq. Ft.", "Sq. Footage", "Volume Band"] if c in stores.columns]
    fig = px.scatter_geo(
        stores,
        lat="Latitude", lon="Longitude", scope="usa", projection="albers usa",
        hover_name="Store Number - Name" if "Store Number - Name" in stores.columns else None,
        hover_data=hover_cols,
        color="State Abbr" if stores["State Abbr"].nunique() > 1 else "Volume Band" if "Volume Band" in stores.columns else None,
        size="24 Volume Dollars" if "24 Volume Dollars" in stores.columns else None,
        title=title,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h"))
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, orientation: str = "v", n: Optional[int] = None) -> go.Figure:
    plot_df = df.copy()
    if n:
        plot_df = plot_df.head(n)
    if orientation == "h":
        fig = px.bar(plot_df, x=x, y=y, orientation="h", title=title)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(plot_df, x=x, y=y, title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def trend_chart(scope_stores: pd.DataFrame, title: str) -> go.Figure:
    rows = []
    for col, year in VOLUME_YEAR_MAP.items():
        cd = col + " Dollars"
        if cd in scope_stores.columns:
            rows.append({"Year": year, "Volume": scope_stores[cd].sum()})
    td = pd.DataFrame(rows)
    fig = px.line(td, x="Year", y="Volume", markers=True, title=title)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), yaxis_tickprefix="$")
    return fig


def display_df(df: pd.DataFrame, money_cols: List[str] = None, pct_cols: List[str] = None, height: int = 360):
    money_cols = money_cols or []
    pct_cols = pct_cols or []
    show = df.copy()
    for c in money_cols:
        if c in show.columns:
            show[c] = show[c].map(fmt_money)
    for c in pct_cols:
        if c in show.columns:
            show[c] = show[c].map(fmt_pct)
    st.dataframe(show, use_container_width=True, height=height)


# -----------------------------
# Sidebar and initial data
# -----------------------------
st.markdown(f'<div class="big-title">🗺️ {APP_NAME}</div>', unsafe_allow_html=True)
st.caption("A practical benchmarking dashboard for store efficiency, growth, saturation, opportunity, category flags, volume bands, and maturity. No AI forecast tab is included.")

pop = load_population()
gdp = load_gdp()

with st.sidebar:
    st.header("Upload Store List")
    uploaded = st.file_uploader("Upload Store List v1.csv", type=["csv"])
    st.markdown("---")
    st.header("Dashboard Controls")
    default_metric = "Store_Count"
    map_metric = st.selectbox(
        "US map metric",
        [
            "Store_Count", "Volume_2024", "Revenue_per_SqFt", "Growth_24_vs_23", "CAGR_21_24",
            "Stores_per_1M_People", "Revenue_per_Capita", "Opportunity_Score", "Avg_Stability", "Avg_Store_Age",
        ],
        index=0,
    )
    top_n = st.slider("Table/chart row limit", 5, 50, 15, 5)

if uploaded is None:
    st.info("Upload `Store List v1.csv` in the sidebar to launch the dashboard. The population and GDP files are already bundled in this GitHub project folder.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bundled file check")
        st.write({
            "Population file found": POP_FILE.exists(),
            "GDP file found": GDP_FILE.exists(),
            "Folder": str(BASE_DIR),
        })
    with col2:
        st.subheader("Expected store file columns")
        st.write(", ".join(["Store Number - Name", "State", "Sq. Footage", "Grand Opening", "21 Volume", "22 Volume", "23 Volume", "24 Volume", "Volume Band"]))
    st.stop()

try:
    raw_stores = read_uploaded_store_file(uploaded)
    stores = normalize_store_df(raw_stores)
    stores = add_store_productivity_scores(stores)
except Exception as e:
    st.error(f"Could not process the uploaded store file: {e}")
    st.stop()

summary = build_state_summary(stores, pop, gdp)
state_options = ["ALL"] + sorted(stores["State Abbr"].dropna().unique().tolist())

with st.sidebar:
    selected_state = st.selectbox(
        "State drilldown",
        state_options,
        format_func=lambda x: "United States / National" if x == "ALL" else f"{ABBR_STATE.get(x, x)} ({x})",
    )

scope = scope_data(stores, selected_state)
selected_label = "United States" if selected_state == "ALL" else ABBR_STATE.get(selected_state, selected_state)
row = state_row(summary, selected_state)

# -----------------------------
# Top KPI strip
# -----------------------------
st.subheader(f"Current View: {selected_label}")
metric_cols = st.columns(6)
metric_cols[0].metric("Stores", fmt_num(len(scope)))
metric_cols[1].metric("2024 Volume", fmt_money(scope["24 Volume Dollars"].sum() if "24 Volume Dollars" in scope.columns else np.nan))
metric_cols[2].metric("Revenue / Sq. Ft.", fmt_money(scope["24 Volume Dollars"].sum() / scope["Sq. Footage"].sum() if "Sq. Footage" in scope.columns and scope["Sq. Footage"].sum() else np.nan))
metric_cols[3].metric("24 vs 23 Growth", fmt_pct((scope["24 Volume Dollars"].sum() - scope["23 Volume Dollars"].sum()) / scope["23 Volume Dollars"].sum() if "23 Volume Dollars" in scope.columns and scope["23 Volume Dollars"].sum() else np.nan))
metric_cols[4].metric("Stores / 1M Pop.", fmt_float(row.get("Stores_per_1M_People", np.nan), 2))
metric_cols[5].metric("Revenue / Capita", fmt_money(row.get("Revenue_per_Capita", np.nan)))

# -----------------------------
# Map explorer outside the 10 metric tabs
# -----------------------------
st.markdown("### Map Explorer")
map_col, loc_col = st.columns([1.05, 1])
with map_col:
    fig = metric_choropleth(summary, map_metric, f"US State Map — {map_metric.replace('_', ' ')}")
    try:
        event = st.plotly_chart(fig, use_container_width=True, key="clickable_state_map", on_select="rerun", selection_mode="points")
        points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
        if points:
            cd = points[0].get("customdata")
            if cd:
                clicked_state = cd[0]
                st.success(f"Map selected: {ABBR_STATE.get(clicked_state, clicked_state)}. Use the sidebar state drilldown to lock this view.")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Your Streamlit version does not expose Plotly click selections. Use the sidebar state drilldown to inspect each state.")
with loc_col:
    st.plotly_chart(store_location_map(scope, f"Store Locations — {selected_label}"), use_container_width=True)

st.markdown("---")

# -----------------------------
# 10 new analytical tabs
# -----------------------------
tabs = st.tabs([
    "1 Revenue / Sq. Ft.",
    "2 YoY Growth",
    "3 Store Density",
    "4 Revenue / Capita",
    "5 Productivity Index",
    "6 Stability",
    "7 Flag Performance",
    "8 Market Opportunity",
    "9 Volume Bands",
    "10 Store Maturity",
])

# 1 Revenue per square foot
with tabs[0]:
    st.markdown("### Revenue per Square Foot")
    st.caption("This separates big-volume stores from truly efficient stores by comparing 2024 volume against physical footprint.")
    c1, c2 = st.columns(2)
    state_eff = summary[summary["Store_Count"].gt(0)].sort_values("Revenue_per_SqFt", ascending=False)
    with c1:
        st.plotly_chart(bar_chart(state_eff.head(top_n), "State", "Revenue_per_SqFt", "Top States by Revenue / Sq. Ft."), use_container_width=True)
    with c2:
        if "Revenue per Sq. Ft." in scope.columns:
            store_eff = scope.sort_values("Revenue per Sq. Ft.", ascending=False)
            st.plotly_chart(bar_chart(store_eff.head(top_n), "Revenue per Sq. Ft.", "Store Number - Name", "Top Stores by Revenue / Sq. Ft.", "h"), use_container_width=True)
    cols = [c for c in ["Store Number - Name", "City", "State", "24 Volume Dollars", "Sq. Footage", "Revenue per Sq. Ft.", "Volume Band"] if c in scope.columns]
    display_df(scope[cols].sort_values("Revenue per Sq. Ft.", ascending=False), money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], height=390)

# 2 YoY Growth
with tabs[1]:
    st.markdown("### Year-over-Year Growth")
    st.caption("Growth metrics are based on actual 2021-2024 store volume history, not forecasts.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(trend_chart(scope, f"Volume Trend — {selected_label}"), use_container_width=True)
    with c2:
        growth_state = summary[summary["Store_Count"].gt(0)].sort_values("Growth_24_vs_23", ascending=False)
        st.plotly_chart(bar_chart(growth_state.head(top_n), "State", "Growth_24_vs_23", "Best 24 vs 23 Growth by State"), use_container_width=True)
    growth_cols = [c for c in ["Store Number - Name", "City", "State", "21 Volume Dollars", "22 Volume Dollars", "23 Volume Dollars", "24 Volume Dollars", "YoY Growth 22 vs 21", "YoY Growth 23 vs 22", "YoY Growth 24 vs 23", "CAGR 21-24"] if c in scope.columns]
    display_df(scope[growth_cols].sort_values("YoY Growth 24 vs 23", ascending=False), money_cols=["21 Volume Dollars", "22 Volume Dollars", "23 Volume Dollars", "24 Volume Dollars"], pct_cols=["YoY Growth 22 vs 21", "YoY Growth 23 vs 22", "YoY Growth 24 vs 23", "CAGR 21-24"], height=410)

# 3 Store Density
with tabs[2]:
    st.markdown("### Store Density by Population")
    st.caption("Useful for identifying underpenetrated or oversaturated states relative to population.")
    c1, c2 = st.columns(2)
    density_df = summary[summary["Store_Count"].gt(0)].copy()
    with c1:
        st.plotly_chart(bar_chart(density_df.sort_values("Stores_per_1M_People", ascending=False).head(top_n), "State", "Stores_per_1M_People", "Highest Stores per 1M People"), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(density_df.sort_values("Population_per_Store", ascending=False).head(top_n), "State", "Population_per_Store", "Highest Population per Store"), use_container_width=True)
    density_cols = ["State", "State Abbr", "Store_Count", "Population 2024", "Stores_per_1M_People", "Population_per_Store", "Density (/mile2)"]
    display_df(density_df[density_cols].sort_values("Stores_per_1M_People", ascending=False), height=410)

# 4 Revenue per Capita
with tabs[3]:
    st.markdown("### Revenue per Capita")
    st.caption("A state-level demand intensity metric: 2024 store volume divided by state population.")
    c1, c2 = st.columns(2)
    rpc = summary[summary["Store_Count"].gt(0)].sort_values("Revenue_per_Capita", ascending=False)
    with c1:
        st.plotly_chart(bar_chart(rpc.head(top_n), "State", "Revenue_per_Capita", "Revenue per Capita by State"), use_container_width=True)
    with c2:
        fig = px.scatter(rpc, x="Population 2024", y="Revenue_per_Capita", size="Store_Count", hover_name="State", title="Population vs Revenue per Capita")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    display_df(rpc[["State", "State Abbr", "Store_Count", "Volume_2024", "Population 2024", "Revenue_per_Capita", "Volume_per_Store"]], money_cols=["Volume_2024", "Revenue_per_Capita", "Volume_per_Store"], height=410)

# 5 Productivity Index
with tabs[4]:
    st.markdown("### Store Productivity Index")
    st.caption("Composite score: 40% revenue/sq.ft., 25% 2024 volume, 20% 2021-2024 CAGR, 15% stability.")
    c1, c2 = st.columns(2)
    prod = scope.sort_values("Productivity Index", ascending=False)
    with c1:
        st.plotly_chart(bar_chart(prod.head(top_n), "Productivity Index", "Store Number - Name", "Top Productivity Stores", "h"), use_container_width=True)
    with c2:
        band_counts = prod["Productivity Band"].value_counts(dropna=False).reset_index()
        band_counts.columns = ["Productivity Band", "Stores"]
        fig = px.pie(band_counts, names="Productivity Band", values="Stores", title="Productivity Band Mix")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    prod_cols = [c for c in ["Store Number - Name", "City", "State", "24 Volume Dollars", "Revenue per Sq. Ft.", "CAGR 21-24", "Stability Score", "Productivity Index", "Productivity Band"] if c in prod.columns]
    display_df(prod[prod_cols], money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], pct_cols=["CAGR 21-24"], height=410)

# 6 Stability
with tabs[5]:
    st.markdown("### Volume Stability")
    st.caption("Stability uses the coefficient of variation across 2021-2024 annual volume. Higher scores are steadier.")
    c1, c2 = st.columns(2)
    stable = scope.sort_values("Stability Score", ascending=False)
    volatile = scope.sort_values("Volume CV", ascending=False)
    with c1:
        st.plotly_chart(bar_chart(stable.head(top_n), "Stability Score", "Store Number - Name", "Most Stable Stores", "h"), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(volatile.head(top_n), "Volume CV", "Store Number - Name", "Most Volatile Stores", "h"), use_container_width=True)
    stab_cols = [c for c in ["Store Number - Name", "City", "State", "Volume Avg 2021-2024", "Volume Std 2021-2024", "Volume CV", "Stability Score", "24 Volume Dollars"] if c in scope.columns]
    display_df(scope[stab_cols].sort_values("Stability Score", ascending=False), money_cols=["Volume Avg 2021-2024", "Volume Std 2021-2024", "24 Volume Dollars"], height=410)

# 7 Flag Performance
with tabs[6]:
    st.markdown("### Category / Market Flag Performance")
    st.caption("Compares stores with each operational or market flag against stores without the flag.")
    rows = []
    base_rev = scope["24 Volume Dollars"].mean() if "24 Volume Dollars" in scope.columns else np.nan
    base_eff = scope["Revenue per Sq. Ft."].mean() if "Revenue per Sq. Ft." in scope.columns else np.nan
    for f in FLAG_COLUMNS:
        fc = f + " Flag"
        if fc in scope.columns:
            yes_df = scope[scope[fc]]
            no_df = scope[~scope[fc]]
            rows.append({
                "Flag": f,
                "Flagged Stores": len(yes_df),
                "Share of Stores": len(yes_df) / len(scope) if len(scope) else np.nan,
                "Avg Volume Flagged": yes_df["24 Volume Dollars"].mean() if not yes_df.empty else np.nan,
                "Avg Volume Non-Flagged": no_df["24 Volume Dollars"].mean() if not no_df.empty else np.nan,
                "Volume Lift vs All": yes_df["24 Volume Dollars"].mean() / base_rev - 1 if not yes_df.empty and base_rev else np.nan,
                "Avg Rev/SqFt Flagged": yes_df["Revenue per Sq. Ft."].mean() if not yes_df.empty and "Revenue per Sq. Ft." in yes_df.columns else np.nan,
                "Rev/SqFt Lift vs All": yes_df["Revenue per Sq. Ft."].mean() / base_eff - 1 if not yes_df.empty and base_eff else np.nan,
                "Avg Growth Flagged": yes_df["YoY Growth 24 vs 23"].mean() if not yes_df.empty and "YoY Growth 24 vs 23" in yes_df.columns else np.nan,
            })
    flag_df = pd.DataFrame(rows).sort_values("Flagged Stores", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(flag_df.head(top_n), "Flag", "Flagged Stores", "Flag Store Counts"), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(flag_df.sort_values("Volume Lift vs All", ascending=False).head(top_n), "Flag", "Volume Lift vs All", "Average Volume Lift vs All Stores"), use_container_width=True)
    display_df(flag_df, money_cols=["Avg Volume Flagged", "Avg Volume Non-Flagged", "Avg Rev/SqFt Flagged"], pct_cols=["Share of Stores", "Volume Lift vs All", "Rev/SqFt Lift vs All", "Avg Growth Flagged"], height=430)

# 8 Market Opportunity
with tabs[7]:
    st.markdown("### Market Opportunity Score")
    st.caption("Scores states using population, GDP, low store density, existing revenue per store, and growth. Higher is more attractive for expansion review.")
    opp = summary[summary["State Abbr"].notna()].copy().sort_values("Opportunity_Score", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(opp.head(top_n), "State", "Opportunity_Score", "Highest Opportunity States"), use_container_width=True)
    with c2:
        fig = px.scatter(opp, x="Stores_per_1M_People", y="Volume_per_Store", size="Population 2024", color="Opportunity_Score", hover_name="State", title="Store Density vs Revenue per Store")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    opp_cols = ["State", "State Abbr", "Store_Count", "Population 2024", "GDP_2020_Millions", "Stores_per_1M_People", "Volume_per_Store", "CAGR_21_24", "Opportunity_Score", "Market_Status"]
    display_df(opp[opp_cols], money_cols=["Volume_per_Store"], pct_cols=["CAGR_21_24"], height=430)

# 9 Volume Bands
with tabs[8]:
    st.markdown("### Volume Band Movement")
    st.caption("Identifies stores close to moving up into the next revenue band or at risk near the bottom of the current band.")
    band_scope = scope.copy()
    c1, c2 = st.columns(2)
    upgrade = band_scope.dropna(subset=["Distance to Next Band"]).sort_values("Distance to Next Band", ascending=True)
    risk = band_scope.dropna(subset=["Distance Above Lower Band"]).sort_values("Distance Above Lower Band", ascending=True)
    with c1:
        st.plotly_chart(bar_chart(upgrade.head(top_n), "Distance to Next Band", "Store Number - Name", "Closest to Next Volume Band", "h"), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(risk.head(top_n), "Distance Above Lower Band", "Store Number - Name", "Closest to Lower Band", "h"), use_container_width=True)
    if "Volume Band" in band_scope.columns:
        band_counts = band_scope.groupby("Volume Band", dropna=False).agg(Stores=("State", "count"), Volume_2024=("24 Volume Dollars", "sum"), Avg_Rev_per_SqFt=("Revenue per Sq. Ft.", "mean")).reset_index()
        st.markdown("#### Volume Band Distribution")
        display_df(band_counts.sort_values("Volume_2024", ascending=False), money_cols=["Volume_2024", "Avg_Rev_per_SqFt"], height=220)
    band_cols = [c for c in ["Store Number - Name", "City", "State", "Volume Band", "24 Volume Dollars", "Band Lower", "Band Upper", "Band Position", "Distance to Next Band", "Distance Above Lower Band"] if c in band_scope.columns]
    display_df(band_scope[band_cols].sort_values("Distance to Next Band", ascending=True), money_cols=["24 Volume Dollars", "Band Lower", "Band Upper", "Distance to Next Band", "Distance Above Lower Band"], height=430)

# 10 Store Maturity
with tabs[9]:
    st.markdown("### Store Age / Maturity Analysis")
    st.caption("Uses Grand Opening to compare new, ramping, mature, and legacy stores.")
    mat = scope.copy()
    c1, c2 = st.columns(2)
    if "Maturity Band" in mat.columns:
        maturity = mat.groupby("Maturity Band", dropna=False).agg(
            Stores=("State", "count"),
            Volume_2024=("24 Volume Dollars", "sum"),
            Avg_Volume=("24 Volume Dollars", "mean"),
            Avg_Rev_per_SqFt=("Revenue per Sq. Ft.", "mean"),
            Avg_Growth=("YoY Growth 24 vs 23", "mean"),
        ).reset_index()
        with c1:
            st.plotly_chart(bar_chart(maturity, "Maturity Band", "Avg_Rev_per_SqFt", "Revenue / Sq. Ft. by Maturity"), use_container_width=True)
        with c2:
            st.plotly_chart(bar_chart(maturity, "Maturity Band", "Avg_Growth", "Growth by Maturity"), use_container_width=True)
        st.markdown("#### Maturity Summary")
        display_df(maturity, money_cols=["Volume_2024", "Avg_Volume", "Avg_Rev_per_SqFt"], pct_cols=["Avg_Growth"], height=240)
    age_cols = [c for c in ["Store Number - Name", "City", "State", "Grand Opening Date", "Grand Opening Year", "Store Age", "Maturity Band", "24 Volume Dollars", "Revenue per Sq. Ft.", "YoY Growth 24 vs 23"] if c in mat.columns]
    display_df(mat[age_cols].sort_values("Store Age", ascending=False), money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], pct_cols=["YoY Growth 24 vs 23"], height=430)

# -----------------------------
# Data audit expander
# -----------------------------
with st.expander("Data audit and raw normalized store data"):
    st.write("Uploaded rows:", len(stores))
    st.write("Volume scale applied:", f"x{stores.attrs.get('volume_scale', 1):,.0f}")
    st.write("Coordinate source:", stores.get("Coordinate Source", pd.Series(dtype=str)).value_counts().to_dict())
    st.write("Columns:", list(stores.columns))
    st.dataframe(stores, use_container_width=True, height=360)
