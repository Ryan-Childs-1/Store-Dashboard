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
        active = summary[summary["Store_Count"].fillna(0).gt(0)].copy()
        if active.empty:
            active = summary.copy()
        active_pop = active["Population 2024"].sum(skipna=True) if "Population 2024" in active.columns else np.nan
        active_gdp = active["GDP_2020_Millions"].sum(skipna=True) if "GDP_2020_Millions" in active.columns else np.nan
        return pd.Series({
            "State": "United States",
            "State Abbr": "ALL",
            "Store_Count": active["Store_Count"].sum(),
            "Total_Sq_Footage": active["Total_Sq_Footage"].sum(),
            "Volume_2024": active["Volume_2024"].sum(),
            "Volume_2023": active["Volume_2023"].sum(),
            "Population 2024": active_pop,
            "GDP_2020_Millions": active_gdp,
            "Revenue_per_SqFt": active["Volume_2024"].sum() / active["Total_Sq_Footage"].sum() if active["Total_Sq_Footage"].sum() else np.nan,
            "Growth_24_vs_23": (active["Volume_2024"].sum() - active["Volume_2023"].sum()) / active["Volume_2023"].sum() if active["Volume_2023"].sum() else np.nan,
            "Stores_per_1M_People": active["Store_Count"].sum() / (active_pop / 1_000_000) if pd.notna(active_pop) and active_pop else np.nan,
            "Revenue_per_Capita": active["Volume_2024"].sum() / active_pop if pd.notna(active_pop) and active_pop else np.nan,
        })
    hit = summary[summary["State Abbr"].eq(selected_state)]
    return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)



def _sorted_clean_options(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    vals = df[col].dropna().map(clean_text)
    vals = vals[vals.ne("") & vals.ne("-")]
    return sorted(vals.unique().tolist())


def _filter_multiselect(df: pd.DataFrame, col: str, selected: List[str]) -> pd.DataFrame:
    if not selected or col not in df.columns:
        return df
    return df[df[col].map(clean_text).isin(selected)].copy()


def render_global_filters(stores_all: pd.DataFrame) -> pd.DataFrame:
    """Sidebar controls that filter the store dataset before every tab is built."""
    filtered = stores_all.copy()
    with st.sidebar:
        st.markdown("---")
        st.header("Global Filters")
        st.caption("These filters apply to the map, state drilldown, side metrics, and every tab.")

        search = st.text_input("Search store / city / address", value="", placeholder="Provo, 102, Utah, address...")
        if search.strip():
            q = search.strip().lower()
            search_cols = [c for c in ["Store Number - Name", "City", "Address", "CityStateZip", "State"] if c in filtered.columns]
            if search_cols:
                mask = pd.Series(False, index=filtered.index)
                for c in search_cols:
                    mask = mask | filtered[c].astype(str).str.lower().str.contains(q, na=False, regex=False)
                filtered = filtered[mask].copy()

        for col, label in [
            ("Region", "Region"),
            ("Size Band", "Size band"),
            ("Volume Band", "Volume band"),
            ("Red/Blue", "Red / Blue"),
            ("Pick Day", "Pick day"),
            ("Maturity Band", "Store maturity"),
            ("Productivity Band", "Productivity band"),
        ]:
            opts = _sorted_clean_options(filtered, col)
            if opts:
                chosen = st.multiselect(label, opts, default=[])
                filtered = _filter_multiselect(filtered, col, chosen)

        numeric_filters = []
        if "24 Volume Dollars" in filtered.columns and filtered["24 Volume Dollars"].notna().any():
            numeric_filters.append(("24 Volume Dollars", "2024 volume"))
        if "Revenue per Sq. Ft." in filtered.columns and filtered["Revenue per Sq. Ft."].notna().any():
            numeric_filters.append(("Revenue per Sq. Ft.", "Revenue / sq. ft."))
        if "Sq. Footage" in filtered.columns and filtered["Sq. Footage"].notna().any():
            numeric_filters.append(("Sq. Footage", "Square footage"))
        if "CAGR 21-24" in filtered.columns and filtered["CAGR 21-24"].notna().any():
            numeric_filters.append(("CAGR 21-24", "2021-2024 CAGR"))

        with st.expander("Numeric filters", expanded=False):
            for col, label in numeric_filters:
                lo = float(pd.to_numeric(filtered[col], errors="coerce").min())
                hi = float(pd.to_numeric(filtered[col], errors="coerce").max())
                if pd.notna(lo) and pd.notna(hi) and lo < hi:
                    sel = st.slider(label, min_value=lo, max_value=hi, value=(lo, hi), key=f"filter_{col}")
                    filtered = filtered[pd.to_numeric(filtered[col], errors="coerce").between(sel[0], sel[1], inclusive="both")].copy()

        available_flags = [f for f in FLAG_COLUMNS if f + " Flag" in filtered.columns]
        if available_flags:
            with st.expander("Category / market flag filters", expanded=False):
                include_flags = st.multiselect("Require these flags", available_flags, default=[])
                exclude_flags = st.multiselect("Exclude these flags", available_flags, default=[])
                mode = st.radio("Flag matching mode", ["All selected flags", "Any selected flag"], horizontal=False)
                if include_flags:
                    cols = [f + " Flag" for f in include_flags]
                    mask = filtered[cols].any(axis=1) if mode == "Any selected flag" else filtered[cols].all(axis=1)
                    filtered = filtered[mask].copy()
                if exclude_flags:
                    cols = [f + " Flag" for f in exclude_flags]
                    filtered = filtered[~filtered[cols].any(axis=1)].copy()

        sort_options = [c for c in [
            "24 Volume Dollars", "Revenue per Sq. Ft.", "YoY Growth 24 vs 23", "CAGR 21-24",
            "Productivity Index", "Stability Score", "Sq. Footage", "Store Age",
        ] if c in filtered.columns]
        if sort_options:
            sort_col = st.selectbox("Default store-table sort", sort_options, index=0)
            sort_dir = st.radio("Sort direction", ["High to low", "Low to high"], horizontal=True)
            filtered = filtered.sort_values(sort_col, ascending=(sort_dir == "Low to high"), na_position="last").copy()

    return filtered


def render_sidebar_metrics(scope_df: pd.DataFrame, summary_df: pd.DataFrame, selected_state: str) -> None:
    """Always-visible side KPI panel tied to the active filters and drilldown."""
    r = state_row(summary_df, selected_state)
    vol_2024 = scope_df["24 Volume Dollars"].sum() if "24 Volume Dollars" in scope_df.columns else np.nan
    vol_2023 = scope_df["23 Volume Dollars"].sum() if "23 Volume Dollars" in scope_df.columns else np.nan
    rev_sqft = vol_2024 / scope_df["Sq. Footage"].sum() if "Sq. Footage" in scope_df.columns and scope_df["Sq. Footage"].sum() else np.nan
    growth = (vol_2024 - vol_2023) / vol_2023 if pd.notna(vol_2023) and vol_2023 else np.nan
    with st.sidebar:
        st.markdown("---")
        st.header("Side Metrics")
        st.caption("Current filters + current drilldown.")
        st.metric("Stores", fmt_num(len(scope_df)))
        st.metric("2024 Volume", fmt_money(vol_2024))
        st.metric("Revenue / Sq. Ft.", fmt_money(rev_sqft))
        st.metric("24 vs 23 Growth", fmt_pct(growth))
        st.metric("Stores / 1M Pop.", fmt_float(r.get("Stores_per_1M_People", np.nan), 2))
        st.metric("Revenue / Capita", fmt_money(r.get("Revenue_per_Capita", np.nan)))
        if "Productivity Index" in scope_df.columns:
            st.metric("Avg Productivity", fmt_float(scope_df["Productivity Index"].mean(), 1))
        if "Stability Score" in scope_df.columns:
            st.metric("Avg Stability", fmt_float(scope_df["Stability Score"].mean(), 1))


# -----------------------------
# Plot helpers
# -----------------------------
def metric_choropleth(summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    # Important: keep the choropleth store count tied to the actual state summary,
    # not to the marker size used for clickable state labels.
    plot_df = summary.copy()
    if "Store_Count" in plot_df.columns:
        plot_df["Store_Count"] = pd.to_numeric(plot_df["Store_Count"], errors="coerce").fillna(0).astype(int)

    hover_fmt = ":,.0f" if metric == "Store_Count" else ":,.2f"
    fig = px.choropleth(
        plot_df,
        locations="State Abbr",
        locationmode="USA-states",
        color=metric,
        scope="usa",
        hover_name="State",
        hover_data={
            "State Abbr": True,
            "Store_Count": ":,.0f" if "Store_Count" in plot_df.columns else False,
            "Volume_2024": ":,.0f" if "Volume_2024" in plot_df.columns else False,
            metric: hover_fmt,
        },
        title=title,
    )

    # Add clickable centroid markers. The marker size is only a visual cue;
    # the hover label now shows the actual store count from customdata.
    map_points = plot_df.dropna(subset=["State Abbr"]).copy()
    map_points["lat"] = map_points["State Abbr"].map(lambda a: STATE_CENTROIDS.get(a, (np.nan, np.nan))[0])
    map_points["lon"] = map_points["State Abbr"].map(lambda a: STATE_CENTROIDS.get(a, (np.nan, np.nan))[1])
    map_points = map_points.dropna(subset=["lat", "lon"])
    map_points["Store_Count"] = pd.to_numeric(map_points.get("Store_Count", 0), errors="coerce").fillna(0).astype(int)
    map_points["Marker_Size"] = np.clip(map_points["Store_Count"] * 3 + 9, 9, 30)
    map_points["State_Label"] = np.where(
        metric == "Store_Count",
        map_points["State Abbr"].astype(str) + "<br>" + map_points["Store_Count"].astype(str),
        map_points["State Abbr"].astype(str),
    )

    fig.add_trace(go.Scattergeo(
        lat=map_points["lat"], lon=map_points["lon"], mode="markers+text",
        text=map_points["State_Label"], textposition="middle center",
        marker=dict(size=map_points["Marker_Size"], color="rgba(20,20,20,.35)", line=dict(width=1, color="white")),
        customdata=map_points[["State Abbr", "Store_Count"]].to_numpy(),
        hovertemplate="%{customdata[0]}<br>Stores: %{customdata[1]:,.0f}<extra>Click/select to drill in</extra>",
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



def winsorize_series(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    """Clip extreme values while preserving missing values."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 4:
        return s
    return s.clip(lower=s.quantile(lo), upper=s.quantile(hi))


def ridge_predict(train_df: pd.DataFrame, feature_cols: List[str], target_col: str, score_df: pd.DataFrame, alpha: float = 2.5) -> Tuple[pd.Series, pd.DataFrame]:
    """Small-sample-safe ridge regression implemented directly with NumPy.

    This avoids adding scikit-learn as a dependency while still creating a real
    regularized model. Missing features are imputed to the training median.
    """
    train = train_df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).copy()
    train = train.dropna(subset=[target_col])
    diagnostics = pd.DataFrame(columns=["Feature", "Coefficient"])
    if len(train) < 4 or len(feature_cols) == 0:
        fallback = pd.to_numeric(score_df.get(target_col), errors="coerce") if target_col in score_df.columns else pd.Series(np.nan, index=score_df.index)
        return fallback.fillna(fallback.median() if fallback.notna().any() else 0.0), diagnostics

    med = train[feature_cols].median(numeric_only=True).fillna(0.0)
    X = train[feature_cols].fillna(med).to_numpy(dtype=float)
    y = train[target_col].to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xz = (X - mu) / sd
    X_design = np.c_[np.ones(len(Xz)), Xz]

    # Penalize feature coefficients, not the intercept.
    penalty = np.eye(X_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(X_design.T @ X_design + penalty) @ X_design.T @ y

    Xs = score_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
    Xsz = (Xs - mu) / sd
    pred = np.c_[np.ones(len(Xsz)), Xsz] @ beta
    diagnostics = pd.DataFrame({"Feature": ["Intercept"] + feature_cols, "Coefficient": beta})
    return pd.Series(pred, index=score_df.index), diagnostics


def build_macro_state_features(summary: pd.DataFrame, pop: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
    """Create state-level macro features from bundled population and GDP files."""
    macro = summary.copy()
    macro["Population_Growth_23_24"] = macro.get("Growth Rate", np.nan)
    if "Population 2023" in macro.columns and "Population 2024" in macro.columns:
        pop_growth_calc = (macro["Population 2024"] - macro["Population 2023"]) / macro["Population 2023"].replace(0, np.nan)
        macro["Population_Growth_23_24"] = macro["Population_Growth_23_24"].fillna(pop_growth_calc)

    gdp_total = pd.DataFrame()
    if not gdp.empty and "Description" in gdp.columns:
        gdp_total = gdp[gdp["Description"].eq("All industry total")].copy()
    year_cols = sorted([c for c in gdp_total.columns if re.fullmatch(r"\d{4}", str(c))])
    if not gdp_total.empty and year_cols:
        rows = []
        for _, r in gdp_total.iterrows():
            abbr = r.get("State Abbr")
            vals = pd.to_numeric(r[year_cols], errors="coerce")
            valid = vals.dropna()
            out = {"State Abbr": abbr}
            if len(valid) >= 2:
                start_year = int(valid.index[0])
                end_year = int(valid.index[-1])
                n = max(end_year - start_year, 1)
                out["GDP_CAGR_All"] = (valid.iloc[-1] / valid.iloc[0]) ** (1 / n) - 1 if valid.iloc[0] > 0 else np.nan
            for span in [5, 10, 15]:
                subset_cols = [c for c in year_cols if int(c) >= int(year_cols[-1]) - span]
                sub = pd.to_numeric(r[subset_cols], errors="coerce").dropna()
                if len(sub) >= 2:
                    n = max(int(sub.index[-1]) - int(sub.index[0]), 1)
                    out[f"GDP_CAGR_{span}Y"] = (sub.iloc[-1] / sub.iloc[0]) ** (1 / n) - 1 if sub.iloc[0] > 0 else np.nan
                    out[f"GDP_Volatility_{span}Y"] = sub.pct_change().replace([np.inf, -np.inf], np.nan).std()
            if len(valid) >= 2:
                out["GDP_Last_Growth"] = valid.pct_change().replace([np.inf, -np.inf], np.nan).iloc[-1]
            rows.append(out)
        gdp_features = pd.DataFrame(rows).dropna(subset=["State Abbr"])
        macro = macro.merge(gdp_features, on="State Abbr", how="left")

    macro["Macro_Growth_Signal"] = (
        0.50 * macro.get("GDP_CAGR_10Y", pd.Series(np.nan, index=macro.index)) +
        0.30 * macro.get("GDP_CAGR_All", pd.Series(np.nan, index=macro.index)) +
        0.20 * macro.get("Population_Growth_23_24", pd.Series(np.nan, index=macro.index))
    )
    return macro


def build_growth_projection_model(stores: pd.DataFrame, summary: pd.DataFrame, pop: pd.DataFrame, gdp: pd.DataFrame, horizon_years: int = 6, scenario: str = "Base") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Advanced but deployment-safe growth model.

    Model design:
    1) Builds macro state features from GDP history and population growth.
    2) Fits a regularized cross-state ridge model to explain 2021-2024 store CAGR.
    3) Blends model-implied growth with recent state/store momentum and national mean reversion.
    4) Projects state and store volumes forward with uncertainty bands.
    """
    macro = build_macro_state_features(summary, pop, gdp)
    work = macro[macro["Store_Count"].fillna(0).gt(0)].copy()
    work["State_CAGR_21_24_W"] = winsorize_series(work["CAGR_21_24"], 0.05, 0.95)
    work["State_Growth_24_23_W"] = winsorize_series(work["Growth_24_vs_23"], 0.05, 0.95)
    work["GDP_CAGR_10Y"] = work.get("GDP_CAGR_10Y", pd.Series(np.nan, index=work.index)).fillna(work.get("GDP_CAGR_All", np.nan))
    work["GDP_CAGR_10Y"] = winsorize_series(work["GDP_CAGR_10Y"], 0.05, 0.95)
    work["Population_Growth_23_24"] = winsorize_series(work.get("Population_Growth_23_24", pd.Series(np.nan, index=work.index)), 0.05, 0.95)
    work["GDP_Volatility_10Y"] = work.get("GDP_Volatility_10Y", pd.Series(np.nan, index=work.index)).fillna(work.get("GDP_Volatility_5Y", np.nan))

    feature_cols = [
        "GDP_CAGR_10Y", "GDP_CAGR_All", "GDP_Volatility_10Y", "Population_Growth_23_24",
        "Stores_per_1M_People", "Revenue_per_Capita", "Revenue_per_GDP_Million", "Volume_per_Store",
        "Revenue_per_SqFt", "Avg_Stability", "Avg_Store_Age", "Store_Count",
    ]
    feature_cols = [c for c in feature_cols if c in work.columns]
    work["Target_CAGR"] = work["State_CAGR_21_24_W"]
    macro_pred, coef_df = ridge_predict(work, feature_cols, "Target_CAGR", work, alpha=3.0)
    work["Macro_Model_CAGR"] = winsorize_series(macro_pred, 0.05, 0.95)

    national_cagr = (stores["24 Volume Dollars"].sum() / stores["21 Volume Dollars"].sum()) ** (1/3) - 1 if stores["21 Volume Dollars"].sum() else work["Target_CAGR"].median()
    national_last = (stores["24 Volume Dollars"].sum() - stores["23 Volume Dollars"].sum()) / stores["23 Volume Dollars"].sum() if stores["23 Volume Dollars"].sum() else national_cagr
    work["Store_Count_Confidence"] = np.sqrt(work["Store_Count"].clip(lower=1)) / np.sqrt(work["Store_Count"].clip(lower=1).max())

    # Hierarchical blend: recent measured state behavior dominates, macro explains persistence, national trend adds mean reversion.
    work["Base_Projected_Growth"] = (
        0.34 * work["State_CAGR_21_24_W"].fillna(national_cagr) +
        0.18 * work["State_Growth_24_23_W"].fillna(national_last) +
        0.28 * work["Macro_Model_CAGR"].fillna(national_cagr) +
        0.12 * work["Macro_Growth_Signal"].fillna(national_cagr) +
        0.08 * national_cagr
    )
    # Low-store states are pulled toward national trend because small sample growth can be noisy.
    work["Base_Projected_Growth"] = work["Base_Projected_Growth"] * work["Store_Count_Confidence"] + national_cagr * (1 - work["Store_Count_Confidence"])
    work["Base_Projected_Growth"] = work["Base_Projected_Growth"].clip(lower=-0.12, upper=0.16)

    scenario_adj = {"Conservative": -0.015, "Base": 0.0, "Optimistic": 0.015}.get(scenario, 0.0)
    work["Projected_Growth_Rate"] = (work["Base_Projected_Growth"] + scenario_adj).clip(lower=-0.15, upper=0.20)
    work["Macro_Projected_Growth_Rate"] = work["Macro_Growth_Signal"].fillna(work["Projected_Growth_Rate"]).clip(lower=-0.08, upper=0.12)
    work["Model_Residual"] = work["Target_CAGR"] - work["Macro_Model_CAGR"]
    residual_sd = work["Model_Residual"].std(skipna=True)
    residual_sd = residual_sd if pd.notna(residual_sd) and residual_sd > 0 else 0.035
    work["Uncertainty"] = (
        residual_sd +
        work[["Growth_22_vs_21", "Growth_23_vs_22", "Growth_24_vs_23"]].std(axis=1, skipna=True).fillna(0.025) * 0.55 +
        (1 - work["Store_Count_Confidence"]) * 0.025
    ).clip(lower=0.02, upper=0.12)

    projection_rows = []
    years = list(range(2024, 2024 + horizon_years + 1))
    for _, r in work.iterrows():
        base_vol = r.get("Volume_2024", np.nan)
        base_pop = r.get("Population 2024", np.nan)
        base_gdp = r.get("GDP_2020_Millions", np.nan)
        # Bring GDP base from 2020 to 2024 with long-run state growth before projecting.
        gdp_base_2024 = base_gdp * ((1 + (r.get("GDP_CAGR_10Y", np.nan) if pd.notna(r.get("GDP_CAGR_10Y", np.nan)) else 0.025)) ** 4) if pd.notna(base_gdp) else np.nan
        for y in years:
            t = y - 2024
            growth = r["Projected_Growth_Rate"]
            macro_growth = r["Macro_Projected_Growth_Rate"]
            unc = r["Uncertainty"] * math.sqrt(max(t, 1))
            projection_rows.append({
                "State": r["State"], "State Abbr": r["State Abbr"], "Year": y,
                "Projected Volume": base_vol * ((1 + growth) ** t) if pd.notna(base_vol) else np.nan,
                "Low Case Volume": base_vol * ((1 + growth - unc) ** t) if pd.notna(base_vol) else np.nan,
                "High Case Volume": base_vol * ((1 + growth + unc) ** t) if pd.notna(base_vol) else np.nan,
                "Projected Population": base_pop * ((1 + r.get("Population_Growth_23_24", 0.0)) ** t) if pd.notna(base_pop) else np.nan,
                "Projected GDP Millions": gdp_base_2024 * ((1 + macro_growth) ** t) if pd.notna(gdp_base_2024) else np.nan,
                "Projected Growth Rate": growth,
                "Macro Growth Signal": macro_growth,
                "Uncertainty": r["Uncertainty"],
            })
    projection = pd.DataFrame(projection_rows)

    store_work = stores.copy()
    state_growth_map = work.set_index("State Abbr")["Projected_Growth_Rate"].to_dict()
    state_uncert_map = work.set_index("State Abbr")["Uncertainty"].to_dict()
    state_macro_map = work.set_index("State Abbr")["Macro_Projected_Growth_Rate"].to_dict()
    store_work["State Projected Growth"] = store_work["State Abbr"].map(state_growth_map).fillna(national_cagr)
    store_work["State Macro Signal"] = store_work["State Abbr"].map(state_macro_map).fillna(national_cagr)
    store_work["Projection Uncertainty"] = store_work["State Abbr"].map(state_uncert_map).fillna(residual_sd)
    productivity_center = store_work["Productivity Index"].median() if "Productivity Index" in store_work.columns else 50.0
    productivity_lift = ((store_work.get("Productivity Index", productivity_center) - productivity_center) / 1000).clip(-0.025, 0.025)
    store_work["Store Momentum CAGR"] = winsorize_series(store_work.get("CAGR 21-24", pd.Series(np.nan, index=store_work.index)), 0.03, 0.97).fillna(store_work["State Projected Growth"])
    store_work["Last YoY Growth"] = winsorize_series(store_work.get("YoY Growth 24 vs 23", pd.Series(np.nan, index=store_work.index)), 0.03, 0.97).fillna(store_work["Store Momentum CAGR"])
    store_work["Projected Store Growth"] = (
        0.40 * store_work["Store Momentum CAGR"] +
        0.35 * store_work["State Projected Growth"] +
        0.15 * store_work["Last YoY Growth"] +
        0.10 * store_work["State Macro Signal"] +
        productivity_lift
    ).clip(lower=-0.18, upper=0.22)
    target_year = 2024 + horizon_years
    store_work[f"Projected {target_year} Volume"] = store_work["24 Volume Dollars"] * ((1 + store_work["Projected Store Growth"]) ** horizon_years)
    store_work[f"Projected {target_year} Low"] = store_work["24 Volume Dollars"] * ((1 + store_work["Projected Store Growth"] - store_work["Projection Uncertainty"]) ** horizon_years)
    store_work[f"Projected {target_year} High"] = store_work["24 Volume Dollars"] * ((1 + store_work["Projected Store Growth"] + store_work["Projection Uncertainty"]) ** horizon_years)
    store_work["Projected Volume Change"] = store_work[f"Projected {target_year} Volume"] - store_work["24 Volume Dollars"]

    work[f"Projected_{target_year}_Volume"] = work["Volume_2024"] * ((1 + work["Projected_Growth_Rate"]) ** horizon_years)
    work[f"Projected_{target_year}_Change"] = work[f"Projected_{target_year}_Volume"] - work["Volume_2024"]
    work["Model_Confidence"] = (100 - work["Uncertainty"] * 450).clip(lower=35, upper=95)
    return work, projection, store_work, coef_df

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
# Strategic insight helpers
# -----------------------------
IMPORTANT_GDP_SEGMENTS = {
    "Retail trade": {"weight": 0.30, "role": "Best direct proxy for the size of the state retail economy."},
    "Accommodation and food services": {"weight": 0.20, "role": "Tourism, travel, camping-trip, and destination-market proxy."},
    "Arts, entertainment, and recreation": {"weight": 0.15, "role": "Closest GDP category to recreation activity."},
    "Agriculture, forestry, fishing and hunting": {"weight": 0.15, "role": "Rural, hunting, fishing, forestry, and outdoor-culture proxy."},
    "Natural resources and mining": {"weight": 0.10, "role": "Rugged/resource-state lifestyle and rural economy proxy."},
    "All industry total": {"weight": 0.10, "role": "General economic strength control."},
}


def get_gdp_segment_panel(gdp: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Return one row per state with selected GDP segments, per-capita values, and segment CAGRs."""
    if gdp.empty or "Description" not in gdp.columns:
        return pd.DataFrame()
    year_cols = sorted([c for c in gdp.columns if re.fullmatch(r"\d{4}", str(c))])
    if not year_cols:
        return pd.DataFrame()
    latest_year = year_cols[-1]
    base_year = "2010" if "2010" in year_cols else year_cols[0]
    rows = []
    for seg in IMPORTANT_GDP_SEGMENTS:
        seg_df = gdp[gdp["Description"].eq(seg)].copy()
        if seg_df.empty:
            continue
        for _, r in seg_df.iterrows():
            abbr = r.get("State Abbr")
            if not isinstance(abbr, str) or not abbr:
                continue
            latest = pd.to_numeric(pd.Series([r.get(latest_year)]), errors="coerce").iloc[0]
            base = pd.to_numeric(pd.Series([r.get(base_year)]), errors="coerce").iloc[0]
            n = max(int(latest_year) - int(base_year), 1)
            cagr = (latest / base) ** (1 / n) - 1 if pd.notna(latest) and pd.notna(base) and base > 0 else np.nan
            rows.append({"State Abbr": abbr, "State": ABBR_STATE.get(abbr, r.get("State")), "Segment": seg, "GDP_Millions": latest, "GDP_CAGR": cagr})
    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame()
    panel = long.pivot_table(index=["State Abbr", "State"], columns="Segment", values="GDP_Millions", aggfunc="first").reset_index()
    cagr_panel = long.pivot_table(index=["State Abbr"], columns="Segment", values="GDP_CAGR", aggfunc="first").reset_index()
    cagr_panel = cagr_panel.rename(columns={c: f"{c} CAGR" for c in cagr_panel.columns if c != "State Abbr"})
    panel = panel.merge(cagr_panel, on="State Abbr", how="left")
    if not pop.empty and "State Abbr" in pop.columns:
        panel = panel.merge(pop[[c for c in ["State Abbr", "Population 2024", "Population 2023", "Growth Rate"] if c in pop.columns]].drop_duplicates("State Abbr"), on="State Abbr", how="left")
    for seg in IMPORTANT_GDP_SEGMENTS:
        if seg in panel.columns:
            panel[f"{seg} per Capita"] = (panel[seg] * 1_000_000) / panel.get("Population 2024", np.nan).replace(0, np.nan)
    return panel


def add_outdoor_macro_scores(summary: pd.DataFrame, gdp: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    panel = get_gdp_segment_panel(gdp, pop)
    if panel.empty:
        out["Outdoor_Retail_Macro_Score"] = np.nan
        out["Retail_GDP_per_Capita"] = np.nan
        out["Outdoor_Macro_GDP_Millions"] = np.nan
        return out
    keep = ["State Abbr"]
    for seg in IMPORTANT_GDP_SEGMENTS:
        keep += [c for c in [seg, f"{seg} CAGR", f"{seg} per Capita"] if c in panel.columns]
    out = out.merge(panel[keep].drop_duplicates("State Abbr"), on="State Abbr", how="left")
    score = pd.Series(0.0, index=out.index)
    weight_sum = 0.0
    macro_total = pd.Series(0.0, index=out.index)
    for seg, meta in IMPORTANT_GDP_SEGMENTS.items():
        per_cap = f"{seg} per Capita"
        if per_cap in out.columns:
            w = meta["weight"]
            score += w * rank_pct(out[per_cap], True).fillna(50)
            weight_sum += w
        if seg in out.columns and seg != "All industry total":
            macro_total = macro_total.add(out[seg].fillna(0) * meta["weight"], fill_value=0)
    out["Outdoor_Retail_Macro_Score"] = score / weight_sum if weight_sum else np.nan
    out["Retail_GDP_per_Capita"] = out.get("Retail trade per Capita", np.nan)
    out["Outdoor_Macro_GDP_Millions"] = macro_total.replace(0, np.nan)
    out["Stores_per_$10B_Retail_GDP"] = out["Store_Count"] / (out.get("Retail trade", np.nan) / 10_000).replace(0, np.nan)
    out["Retail_GDP_per_Store_Millions"] = out.get("Retail trade", np.nan) / out["Store_Count"].replace(0, np.nan)
    out["Outdoor_Macro_per_Store_Millions"] = out["Outdoor_Macro_GDP_Millions"] / out["Store_Count"].replace(0, np.nan)
    out["Market_Penetration_Retail_GDP"] = out["Volume_2024"] / (out.get("Retail trade", np.nan) * 1_000_000).replace(0, np.nan)
    out["Outdoor_Market_Penetration"] = out["Volume_2024"] / (out["Outdoor_Macro_GDP_Millions"] * 1_000_000).replace(0, np.nan)
    out["White_Space_Score"] = (
        0.32 * rank_pct(out.get("Outdoor_Retail_Macro_Score"), True).fillna(50) +
        0.22 * rank_pct(out.get("Population 2024"), True).fillna(50) +
        0.18 * rank_pct(out.get("Retail_GDP_per_Store_Millions"), True).fillna(50) +
        0.18 * rank_pct(out.get("Stores_per_1M_People"), False).fillna(50) +
        0.10 * rank_pct(out.get("Volume_per_Store"), True).fillna(50)
    )
    out["Risk_Score"] = (
        0.35 * rank_pct(out.get("Growth_24_vs_23"), False).fillna(50) +
        0.25 * rank_pct(out.get("CAGR_21_24"), False).fillna(50) +
        0.20 * rank_pct(out.get("Revenue_per_SqFt"), False).fillna(50) +
        0.20 * rank_pct(out.get("Store_Count"), True).fillna(50)
    )
    # Four-quadrant strategic classification.
    macro_med = out["Outdoor_Retail_Macro_Score"].median(skipna=True)
    density_med = out["Stores_per_1M_People"].median(skipna=True)
    def _fit_status(r):
        strong = r.get("Outdoor_Retail_Macro_Score", np.nan) >= macro_med
        saturated = r.get("Stores_per_1M_People", np.nan) >= density_med
        if strong and not saturated:
            return "Strong Fit / Underpenetrated"
        if strong and saturated:
            return "Strong Fit / Saturated"
        if (not strong) and saturated:
            return "Weak Fit / Overstored"
        return "Weak Fit / Low Presence"
    out["Outdoor_Fit_Status"] = out.apply(_fit_status, axis=1)
    return out


def add_store_diagnostics(stores: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    df = stores.copy()
    state_map = summary.set_index("State Abbr") if "State Abbr" in summary.columns else pd.DataFrame()
    if not state_map.empty:
        for c in ["Outdoor_Retail_Macro_Score", "White_Space_Score", "Risk_Score", "Revenue_per_SqFt", "Growth_24_vs_23", "Stores_per_1M_People", "Retail_GDP_per_Store_Millions", "Outdoor_Fit_Status"]:
            if c in state_map.columns:
                df[f"State {c}"] = df["State Abbr"].map(state_map[c])
    # Expected productivity by peer group: state + size band + maturity band blend.
    df["Peer Expected Rev/SqFt"] = np.nan
    components = []
    for col, wt in [("State Abbr", 0.35), ("Size Band", 0.30), ("Maturity Band", 0.20), ("Region", 0.15)]:
        if col in df.columns and "Revenue per Sq. Ft." in df.columns:
            m = df.groupby(col, dropna=False)["Revenue per Sq. Ft."].transform("median")
            components.append((m, wt))
    if components:
        num = sum(s.fillna(df["Revenue per Sq. Ft."].median()) * wt for s, wt in components)
        den = sum(wt for _, wt in components)
        df["Peer Expected Rev/SqFt"] = num / den
    if "Revenue per Sq. Ft." in df.columns:
        df["Rev/SqFt Gap"] = df["Revenue per Sq. Ft."] - df["Peer Expected Rev/SqFt"]
        df["Rev/SqFt Gap %"] = df["Rev/SqFt Gap"] / df["Peer Expected Rev/SqFt"].replace(0, np.nan)
    df["Store Diagnostic"] = "Needs deeper review"
    if "Store Age" in df.columns:
        df.loc[df["Store Age"].fillna(99).lt(3), "Store Diagnostic"] = "Ramp-up store: compare carefully"
    if "Rev/SqFt Gap %" in df.columns:
        df.loc[df["Rev/SqFt Gap %"].lt(-0.25) & df.get("Sq. Footage", pd.Series(0, index=df.index)).gt(df.get("Sq. Footage", pd.Series(0, index=df.index)).median()), "Store Diagnostic"] = "Oversized footprint / low productivity"
        df.loc[df["Rev/SqFt Gap %"].lt(-0.25) & df.get("State Risk_Score", pd.Series(0, index=df.index)).gt(65), "Store Diagnostic"] = "Statewide weakness + local underperformance"
        df.loc[df["Rev/SqFt Gap %"].gt(0.25), "Store Diagnostic"] = "Outperformer: study/replicate"
    # Store archetypes.
    high_eff = df.get("Revenue per Sq. Ft.", pd.Series(np.nan, index=df.index)) >= df.get("Revenue per Sq. Ft.", pd.Series(np.nan, index=df.index)).quantile(0.70)
    high_vol = df.get("24 Volume Dollars", pd.Series(np.nan, index=df.index)) >= df.get("24 Volume Dollars", pd.Series(np.nan, index=df.index)).quantile(0.70)
    large = df.get("Sq. Footage", pd.Series(np.nan, index=df.index)) >= df.get("Sq. Footage", pd.Series(np.nan, index=df.index)).quantile(0.70)
    low_eff = df.get("Revenue per Sq. Ft.", pd.Series(np.nan, index=df.index)) <= df.get("Revenue per Sq. Ft.", pd.Series(np.nan, index=df.index)).quantile(0.30)
    df["Store Archetype"] = "Balanced / Core Store"
    df.loc[high_eff & ~large, "Store Archetype"] = "Compact High-Productivity"
    df.loc[large & high_vol, "Store Archetype"] = "Large Destination Store"
    df.loc[large & low_eff, "Store Archetype"] = "Large Underproductive Box"
    if "Metro Flag" in df.columns:
        df.loc[df["Metro Flag"].fillna(False), "Store Archetype"] = "Metro Store"
    for flag, label in [("Ammo Hub Flag", "Ammo Hub Store"), ("Ocean Flag", "Water/Ocean Store"), ("Year Round Kayaks Flag", "Paddlesports Store")]:
        if flag in df.columns:
            df.loc[df[flag].fillna(False), "Store Archetype"] = label
    if "Store Age" in df.columns:
        df.loc[df["Store Age"].fillna(99).lt(3), "Store Archetype"] = "New / Ramp Store"
    return df


def insight_sentence(label: str, value: str, explanation: str) -> None:
    st.markdown(f"**{label}:** {value} — {explanation}")


def safe_col_list(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# -----------------------------
# Sidebar and initial data
# -----------------------------
st.markdown(f'<div class="big-title">🗺️ {APP_NAME}</div>', unsafe_allow_html=True)
st.caption("A practical benchmarking dashboard for store efficiency, growth, saturation, opportunity, category flags, advanced growth modeling, and maturity. The old AI forecasts tab has been removed.")

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
    st.caption("Tables and charts are uncapped: every row passing the global filters is included.")

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
    stores_all = normalize_store_df(raw_stores)
    stores_all = add_store_productivity_scores(stores_all)
except Exception as e:
    st.error(f"Could not process the uploaded store file: {e}")
    st.stop()

stores = render_global_filters(stores_all)
if stores.empty:
    st.warning("No stores match the current global filters. Clear one or more filters in the sidebar to continue.")
    st.stop()

summary = build_state_summary(stores, pop, gdp)
summary = add_outdoor_macro_scores(summary, gdp, pop)
stores = add_store_diagnostics(stores, summary)
stores_all = add_store_diagnostics(stores_all, summary)
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
render_sidebar_metrics(scope, summary, selected_state)

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
        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1138")
        st.caption("Your Streamlit version does not expose Plotly click selections. Use the sidebar state drilldown to inspect each state.")
with loc_col:
    st.plotly_chart(store_location_map(scope, f"Store Locations — {selected_label}"), use_container_width=True, key="plotly_chart_1141")

st.markdown("---")

# -----------------------------
# 10 new analytical tabs
# -----------------------------

tabs = st.tabs([
    "1 Major Insights",
    "2 Revenue / Sq. Ft.",
    "3 Growth & Risk",
    "4 Store Format Efficiency",
    "5 Maturity Cohorts",
    "6 Saturation & White Space",
    "7 Outdoor Market Fit / GDP",
    "8 Market Penetration",
    "9 Productivity Index",
    "10 Flag Performance",
    "11 Store Archetypes",
    "12 Best Practices",
    "13 Underperformer Diagnostic",
    "14 Growth Projection",
    "15 Stability",
])

# 1 Major Insights
with tabs[0]:
    st.markdown("### Store List v1 — Major Insights")
    st.caption("These observations are generated from the uploaded Store List v1 file and filtered by the current sidebar settings.")
    total_24 = scope["24 Volume Dollars"].sum() if "24 Volume Dollars" in scope.columns else np.nan
    total_23 = scope["23 Volume Dollars"].sum() if "23 Volume Dollars" in scope.columns else np.nan
    total_21 = scope["21 Volume Dollars"].sum() if "21 Volume Dollars" in scope.columns else np.nan
    yoy = (total_24 - total_23) / total_23 if pd.notna(total_23) and total_23 else np.nan
    cagr = (total_24 / total_21) ** (1/3) - 1 if pd.notna(total_21) and total_21 else np.nan
    weighted_rpsf = total_24 / scope["Sq. Footage"].sum() if "Sq. Footage" in scope.columns and scope["Sq. Footage"].sum() else np.nan
    insight_cols = st.columns(5)
    insight_cols[0].metric("2024 Volume", fmt_money(total_24))
    insight_cols[1].metric("2024 vs 2023", fmt_pct(yoy))
    insight_cols[2].metric("2021-2024 CAGR", fmt_pct(cagr))
    insight_cols[3].metric("Weighted Rev/SqFt", fmt_money(weighted_rpsf))
    insight_cols[4].metric("Stores", fmt_num(len(scope)))

    st.markdown("#### Executive readout")
    insight_sentence("Network direction", fmt_pct(yoy), "latest-year growth shows whether the current store set is expanding or contracting versus 2023.")
    insight_sentence("Longer-term trend", fmt_pct(cagr), "the 2021-2024 CAGR helps separate one-year noise from multi-year trend pressure.")
    insight_sentence("Footprint productivity", fmt_money(weighted_rpsf), "weighted revenue per square foot is a stronger efficiency measure than total revenue alone.")

    c1, c2 = st.columns(2)
    with c1:
        top_states = summary[summary["Store_Count"].gt(0)].sort_values("Volume_2024", ascending=False)
        st.plotly_chart(bar_chart(top_states, "State", "Volume_2024", "States by 2024 Volume"), use_container_width=True, key="plotly_chart_1192")
    with c2:
        state_eff = summary[summary["Store_Count"].gt(0)].sort_values("Revenue_per_SqFt", ascending=False)
        st.plotly_chart(bar_chart(state_eff, "State", "Revenue_per_SqFt", "States by Weighted Revenue / Sq. Ft."), use_container_width=True, key="plotly_chart_1195")

    st.markdown("#### Automatically surfaced findings")
    finding_rows = []
    if not summary.empty:
        best_eff = summary[summary["Store_Count"].gt(0)].sort_values("Revenue_per_SqFt", ascending=False).head(1)
        worst_growth = summary[summary["Store_Count"].gt(0)].sort_values("Growth_24_vs_23", ascending=True).head(1)
        best_volume = summary[summary["Store_Count"].gt(0)].sort_values("Volume_2024", ascending=False).head(1)
        if not best_eff.empty:
            r = best_eff.iloc[0]
            finding_rows.append({"Insight": "Highest state productivity", "Result": f"{r['State']} at {fmt_money(r['Revenue_per_SqFt'])} per sq. ft.", "Why it matters": "Shows the strongest footprint efficiency."})
        if not worst_growth.empty:
            r = worst_growth.iloc[0]
            finding_rows.append({"Insight": "Largest state decline", "Result": f"{r['State']} at {fmt_pct(r['Growth_24_vs_23'])} YoY", "Why it matters": "Flags a market needing diagnosis."})
        if not best_volume.empty:
            r = best_volume.iloc[0]
            finding_rows.append({"Insight": "Largest revenue state", "Result": f"{r['State']} at {fmt_money(r['Volume_2024'])}", "Why it matters": "Shows where the largest absolute revenue base sits."})
    if "Size Band" in scope.columns:
        band = scope.groupby("Size Band", dropna=False).agg(Stores=("State", "count"), Avg_RevSqFt=("Revenue per Sq. Ft.", "mean"), Volume=("24 Volume Dollars", "sum")).reset_index().sort_values("Avg_RevSqFt", ascending=False)
        if not band.empty:
            r=band.iloc[0]
            finding_rows.append({"Insight":"Best size band efficiency", "Result": f"{clean_text(r['Size Band'])}: {fmt_money(r['Avg_RevSqFt'])} avg rev/sq. ft.", "Why it matters":"Tests whether smaller or larger formats are more efficient."})
    if finding_rows:
        display_df(pd.DataFrame(finding_rows), height=220)

    st.markdown("#### Store-level highlights")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Best practice candidates**")
        cols = safe_col_list(scope, ["Store Number - Name", "City", "State", "24 Volume Dollars", "Revenue per Sq. Ft.", "CAGR 21-24", "Productivity Index", "Store Archetype"])
        display_df(scope.sort_values("Productivity Index", ascending=False)[cols], money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], pct_cols=["CAGR 21-24"], height=320)
    with h2:
        st.markdown("**Risk candidates**")
        risk_cols = safe_col_list(scope, ["Store Number - Name", "City", "State", "24 Volume Dollars", "Revenue per Sq. Ft.", "YoY Growth 24 vs 23", "CAGR 21-24", "Store Diagnostic"])
        risk_sort = "YoY Growth 24 vs 23" if "YoY Growth 24 vs 23" in scope.columns else "Revenue per Sq. Ft."
        display_df(scope.sort_values(risk_sort, ascending=True)[risk_cols], money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], pct_cols=["YoY Growth 24 vs 23", "CAGR 21-24"], height=320)

# 2 Revenue per square foot
with tabs[1]:
    st.markdown("### Revenue per Square Foot")
    st.info("This is the core store-efficiency metric: it separates stores that are simply large from stores that are truly productive.")
    c1, c2 = st.columns(2)
    state_eff = summary[summary["Store_Count"].gt(0)].sort_values("Revenue_per_SqFt", ascending=False)
    with c1:
        st.plotly_chart(bar_chart(state_eff, "State", "Revenue_per_SqFt", "States by Weighted Revenue / Sq. Ft."), use_container_width=True, key="plotly_chart_1239")
    with c2:
        if "Revenue per Sq. Ft." in scope.columns:
            fig = px.scatter(scope, x="Sq. Footage", y="Revenue per Sq. Ft.", size="24 Volume Dollars", color="State Abbr", hover_name="Store Number - Name", title="Footprint vs Productivity")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1244")
    cols = safe_col_list(scope, ["Store Number - Name", "City", "State", "Size Band", "24 Volume Dollars", "Sq. Footage", "Revenue per Sq. Ft.", "Store Diagnostic"])
    display_df(scope[cols].sort_values("Revenue per Sq. Ft.", ascending=False), money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], height=430)

# 3 Growth & Risk
with tabs[2]:
    st.markdown("### Growth & Risk Monitor")
    st.info("This view focuses on actual measured growth/decline. New stores with missing older history should be interpreted separately from mature stores.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(trend_chart(scope, f"Volume Trend — {selected_label}"), use_container_width=True, key="plotly_chart_1254")
    with c2:
        state_risk = summary[summary["Store_Count"].gt(0)].sort_values("Risk_Score", ascending=False)
        st.plotly_chart(bar_chart(state_risk, "State", "Risk_Score", "Highest State Risk Scores"), use_container_width=True, key="plotly_chart_1257")
    risk_cols = safe_col_list(summary, ["State", "Store_Count", "Volume_2024", "Growth_24_vs_23", "CAGR_21_24", "Revenue_per_SqFt", "Risk_Score"])
    display_df(summary[summary["Store_Count"].gt(0)].sort_values("Risk_Score", ascending=False)[risk_cols], money_cols=["Volume_2024", "Revenue_per_SqFt"], pct_cols=["Growth_24_vs_23", "CAGR_21_24"], height=320)
    st.markdown("#### Store decline table")
    cols = safe_col_list(scope, ["Store Number - Name", "City", "State", "24 Volume Dollars", "YoY Growth 24 vs 23", "CAGR 21-24", "Stability Score", "Store Diagnostic"])
    display_df(scope.sort_values("YoY Growth 24 vs 23", ascending=True)[cols], money_cols=["24 Volume Dollars"], pct_cols=["YoY Growth 24 vs 23", "CAGR 21-24"], height=380)

# 4 Store Format Efficiency
with tabs[3]:
    st.markdown("### Store Format Efficiency")
    st.info("This tab tests whether larger stores are earning enough incremental sales to justify their additional footprint.")
    if "Size Band" in scope.columns:
        band = scope.groupby("Size Band", dropna=False).agg(
            Stores=("State", "count"), Volume_2024=("24 Volume Dollars", "sum"), Avg_RevSqFt=("Revenue per Sq. Ft.", "mean"),
            Weighted_RevSqFt=("24 Volume Dollars", "sum"), SqFt=("Sq. Footage", "sum"), Avg_Growth=("YoY Growth 24 vs 23", "mean")
        ).reset_index()
        band["Weighted_RevSqFt"] = band["Volume_2024"] / band["SqFt"].replace(0, np.nan)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bar_chart(band.sort_values("Weighted_RevSqFt", ascending=False), "Size Band", "Weighted_RevSqFt", "Weighted Revenue / Sq. Ft. by Size Band"), use_container_width=True, key="plotly_chart_1276")
        with c2:
            fig = px.box(scope, x="Size Band", y="Revenue per Sq. Ft.", points="all", title="Store Productivity Distribution by Size Band")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1280")
        display_df(band.sort_values("Weighted_RevSqFt", ascending=False), money_cols=["Volume_2024", "Avg_RevSqFt", "Weighted_RevSqFt"], pct_cols=["Avg_Growth"], height=300)
    st.markdown("#### Large footprint risk and compact outperformers")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Large underproductive stores**")
        cols = safe_col_list(scope, ["Store Number - Name", "State", "Size Band", "Sq. Footage", "24 Volume Dollars", "Revenue per Sq. Ft.", "Store Diagnostic"])
        display_df(scope.sort_values(["Sq. Footage", "Revenue per Sq. Ft."], ascending=[False, True])[cols], money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], height=330)
    with c4:
        st.markdown("**Compact high-productivity stores**")
        cols = safe_col_list(scope, ["Store Number - Name", "State", "Size Band", "Sq. Footage", "24 Volume Dollars", "Revenue per Sq. Ft.", "Productivity Index"])
        compact = scope.sort_values(["Revenue per Sq. Ft.", "Sq. Footage"], ascending=[False, True])
        display_df(compact[cols], money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], height=330)

# 5 Maturity Cohorts
with tabs[4]:
    st.markdown("### Mature vs New Store Performance")
    st.info("Growth should be interpreted by store age. New stores often have missing early-year volume, while mature stores give a cleaner read on organic performance.")
    if "Maturity Band" in scope.columns:
        cohort = scope.groupby("Maturity Band", dropna=False).agg(
            Stores=("State", "count"), Volume_2024=("24 Volume Dollars", "sum"), Avg_RevSqFt=("Revenue per Sq. Ft.", "mean"),
            Avg_YoY=("YoY Growth 24 vs 23", "mean"), Avg_CAGR=("CAGR 21-24", "mean"), Avg_Age=("Store Age", "mean")
        ).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bar_chart(cohort, "Maturity Band", "Avg_RevSqFt", "Revenue / Sq. Ft. by Maturity Cohort"), use_container_width=True, key="plotly_chart_1305")
        with c2:
            st.plotly_chart(bar_chart(cohort, "Maturity Band", "Avg_YoY", "Latest YoY Growth by Maturity Cohort"), use_container_width=True, key="plotly_chart_1307")
        display_df(cohort, money_cols=["Volume_2024", "Avg_RevSqFt"], pct_cols=["Avg_YoY", "Avg_CAGR"], height=300)
    cols = safe_col_list(scope, ["Store Number - Name", "State", "Grand Opening Year", "Store Age", "Maturity Band", "24 Volume Dollars", "Revenue per Sq. Ft.", "YoY Growth 24 vs 23", "CAGR 21-24"])
    display_df(scope.sort_values("Store Age", ascending=True)[cols], money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], pct_cols=["YoY Growth 24 vs 23", "CAGR 21-24"], height=380)

# 6 Saturation & White Space
with tabs[5]:
    st.markdown("### State Saturation & White Space")
    st.info("This turns store density into an expansion-oriented view: population, retail GDP, outdoor macro score, current store count, and revenue productivity are considered together.")
    ws = summary[summary["Store_Count"].gt(0)].sort_values("White_Space_Score", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(ws, "State", "White_Space_Score", "White Space Score"), use_container_width=True, key="plotly_chart_1319")
    with c2:
        fig = px.scatter(ws, x="Stores_per_1M_People", y="Retail_GDP_per_Store_Millions", size="Population 2024", color="White_Space_Score", hover_name="State", title="Store Density vs Retail GDP per Store")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1323")
    cols = safe_col_list(ws, ["State", "Store_Count", "Population 2024", "Stores_per_1M_People", "Retail_GDP_per_Store_Millions", "Outdoor_Macro_per_Store_Millions", "Revenue_per_Capita", "White_Space_Score", "Outdoor_Fit_Status"])
    display_df(ws[cols], money_cols=["Revenue_per_Capita"], height=430)

# 7 Outdoor Market Fit / GDP
with tabs[6]:
    st.markdown("### Outdoor Market Fit / GDP Segment Analysis")
    st.info("For a hunting, camping, outdoor, and firearms retailer, total GDP is too broad. The app prioritizes retail trade, tourism-adjacent GDP, recreation GDP, agriculture/forestry/fishing/hunting, natural resources, and total GDP as a control.")
    gdp_panel = get_gdp_segment_panel(gdp, pop)
    gdp_docs = pd.DataFrame([{"GDP Segment": k, "Weight": v["weight"], "Why selected": v["role"]} for k, v in IMPORTANT_GDP_SEGMENTS.items()])
    display_df(gdp_docs, pct_cols=["Weight"], height=230)
    fit = summary[summary["Store_Count"].gt(0)].sort_values("Outdoor_Retail_Macro_Score", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(fit, "State", "Outdoor_Retail_Macro_Score", "Outdoor Retail Macro Score"), use_container_width=True, key="plotly_chart_1337")
    with c2:
        fig = px.scatter(fit, x="Outdoor_Retail_Macro_Score", y="Revenue_per_Capita", size="Store_Count", color="Outdoor_Fit_Status", hover_name="State", title="Macro Fit vs Current Revenue / Capita")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1341")
    cols = safe_col_list(fit, ["State", "Store_Count", "Outdoor_Retail_Macro_Score", "Outdoor_Fit_Status", "Retail_GDP_per_Capita", "Retail trade", "Accommodation and food services", "Arts, entertainment, and recreation", "Agriculture, forestry, fishing and hunting", "Natural resources and mining"])
    display_df(fit[cols], money_cols=["Retail_GDP_per_Capita"], height=430)

# 8 Market Penetration
with tabs[7]:
    st.markdown("### Market Penetration")
    st.info("Revenue per capita is useful, but penetration against relevant retail/outdoor GDP is a more strategic measure of whether a state is over- or under-performing its economic base.")
    pen = summary[summary["Store_Count"].gt(0)].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(pen.sort_values("Market_Penetration_Retail_GDP", ascending=False), "State", "Market_Penetration_Retail_GDP", "Store Revenue / Retail GDP"), use_container_width=True, key="plotly_chart_1352")
    with c2:
        st.plotly_chart(bar_chart(pen.sort_values("Outdoor_Market_Penetration", ascending=False), "State", "Outdoor_Market_Penetration", "Store Revenue / Outdoor Macro GDP"), use_container_width=True, key="plotly_chart_1354")
    cols = safe_col_list(pen, ["State", "Store_Count", "Volume_2024", "Revenue_per_Capita", "Market_Penetration_Retail_GDP", "Outdoor_Market_Penetration", "Retail_GDP_per_Store_Millions", "Outdoor_Macro_per_Store_Millions"])
    display_df(pen.sort_values("Market_Penetration_Retail_GDP", ascending=False)[cols], money_cols=["Volume_2024", "Revenue_per_Capita"], pct_cols=["Market_Penetration_Retail_GDP", "Outdoor_Market_Penetration"], height=430)

# 9 Productivity Index
with tabs[8]:
    st.markdown("### Productivity Index")
    st.info("The default index blends revenue per square foot, 2024 volume, measured growth, and stability. It is designed to find productive stores, not just large ones.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(scope.sort_values("Productivity Index", ascending=False), "Productivity Index", "Store Number - Name", "Store Productivity Leaders", "h"), use_container_width=True, key="plotly_chart_1364")
    with c2:
        if "Productivity Band" in scope.columns:
            band = scope.groupby("Productivity Band", dropna=False).agg(Stores=("State", "count"), Volume_2024=("24 Volume Dollars", "sum"), Avg_RevSqFt=("Revenue per Sq. Ft.", "mean")).reset_index()
            st.plotly_chart(bar_chart(band, "Productivity Band", "Stores", "Stores by Productivity Band"), use_container_width=True, key="plotly_chart_1368")
    cols = safe_col_list(scope, ["Store Number - Name", "City", "State", "24 Volume Dollars", "Revenue per Sq. Ft.", "CAGR 21-24", "Stability Score", "Productivity Index", "Productivity Band", "Store Archetype"])
    display_df(scope.sort_values("Productivity Index", ascending=False)[cols], money_cols=["24 Volume Dollars", "Revenue per Sq. Ft."], pct_cols=["CAGR 21-24"], height=430)

# 10 Flag Performance
with tabs[9]:
    st.markdown("### Flag Performance & Merchandising Impact")
    st.info("This tab compares stores with each operational/market flag against stores without it. Treat small-sample flags carefully.")
    rows=[]
    base_rev = scope["24 Volume Dollars"].mean() if "24 Volume Dollars" in scope.columns else np.nan
    base_eff = scope["Revenue per Sq. Ft."].mean() if "Revenue per Sq. Ft." in scope.columns else np.nan
    for f in FLAG_COLUMNS:
        col=f+" Flag"
        if col in scope.columns:
            yes=scope[scope[col].fillna(False)]
            no=scope[~scope[col].fillna(False)]
            if len(yes)>0:
                rows.append({"Flag":f,"Flagged Stores":len(yes),"Share of Stores":len(yes)/len(scope),"Avg Volume Flagged":yes["24 Volume Dollars"].mean(),"Avg Volume Non-Flagged":no["24 Volume Dollars"].mean() if len(no) else np.nan,"Volume Lift vs All":yes["24 Volume Dollars"].mean()/base_rev-1 if base_rev else np.nan,"Avg Rev/SqFt Flagged":yes["Revenue per Sq. Ft."].mean(),"Rev/SqFt Lift vs All":yes["Revenue per Sq. Ft."].mean()/base_eff-1 if base_eff else np.nan,"Avg Growth Flagged":yes.get("YoY Growth 24 vs 23", pd.Series(dtype=float)).mean()})
    flag_df=pd.DataFrame(rows).sort_values("Flagged Stores", ascending=False) if rows else pd.DataFrame()
    if not flag_df.empty:
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(bar_chart(flag_df, "Flag", "Flagged Stores", "Flag Store Counts"), use_container_width=True, key="plotly_chart_1390")
        with c2:
            st.plotly_chart(bar_chart(flag_df.sort_values("Rev/SqFt Lift vs All", ascending=False), "Flag", "Rev/SqFt Lift vs All", "Revenue / SqFt Lift vs All Stores"), use_container_width=True, key="plotly_chart_1392")
        display_df(flag_df, money_cols=["Avg Volume Flagged","Avg Volume Non-Flagged","Avg Rev/SqFt Flagged"], pct_cols=["Share of Stores","Volume Lift vs All","Rev/SqFt Lift vs All","Avg Growth Flagged"], height=430)

# 11 Store Archetypes
with tabs[10]:
    st.markdown("### Store Archetype Clustering")
    st.info("Stores should be compared against similar stores. This tab classifies stores by format, productivity, market flags, and maturity.")
    if "Store Archetype" in scope.columns:
        arch = scope.groupby("Store Archetype", dropna=False).agg(Stores=("State", "count"), Volume_2024=("24 Volume Dollars", "sum"), Avg_RevSqFt=("Revenue per Sq. Ft.", "mean"), Avg_Growth=("YoY Growth 24 vs 23", "mean"), Avg_Productivity=("Productivity Index", "mean")).reset_index().sort_values("Avg_Productivity", ascending=False)
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(bar_chart(arch, "Store Archetype", "Avg_Productivity", "Average Productivity by Archetype"), use_container_width=True, key="plotly_chart_1403")
        with c2:
            st.plotly_chart(bar_chart(arch, "Store Archetype", "Avg_RevSqFt", "Revenue / Sq. Ft. by Archetype"), use_container_width=True, key="plotly_chart_1405")
        display_df(arch, money_cols=["Volume_2024","Avg_RevSqFt"], pct_cols=["Avg_Growth"], height=300)
    cols=safe_col_list(scope,["Store Number - Name","State","Size Band","Maturity Band","Store Archetype","24 Volume Dollars","Revenue per Sq. Ft.","Productivity Index","Store Diagnostic"])
    display_df(scope.sort_values("Store Archetype")[cols], money_cols=["24 Volume Dollars","Revenue per Sq. Ft."], height=380)

# 12 Best Practices
with tabs[11]:
    st.markdown("### Best Practices / Clone These Stores")
    st.info("These are stores worth studying before new openings, remodels, or category changes because they combine volume, efficiency, stability, and market fit.")
    clone = scope.copy()
    clone["Clone Score"] = (
        0.35*rank_pct(clone.get("Revenue per Sq. Ft."), True).fillna(50)+
        0.20*rank_pct(clone.get("24 Volume Dollars"), True).fillna(50)+
        0.15*rank_pct(clone.get("Stability Score"), True).fillna(50)+
        0.15*rank_pct(clone.get("Productivity Index"), True).fillna(50)+
        0.15*rank_pct(clone.get("State Outdoor_Retail_Macro_Score"), True).fillna(50)
    )
    c1,c2=st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(clone.sort_values("Clone Score", ascending=False), "Clone Score", "Store Number - Name", "Clone Score Leaders", "h"), use_container_width=True, key="plotly_chart_1424")
    with c2:
        fig=px.scatter(clone, x="Revenue per Sq. Ft.", y="YoY Growth 24 vs 23", size="24 Volume Dollars", color="Store Archetype", hover_name="Store Number - Name", title="Efficiency + Momentum Best-Practice Map")
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10), xaxis_tickprefix="$", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1428")
    cols=safe_col_list(clone,["Store Number - Name","City","State","Store Archetype","24 Volume Dollars","Revenue per Sq. Ft.","YoY Growth 24 vs 23","Stability Score","Productivity Index","Clone Score"])
    display_df(clone.sort_values("Clone Score", ascending=False)[cols], money_cols=["24 Volume Dollars","Revenue per Sq. Ft."], pct_cols=["YoY Growth 24 vs 23"], height=430)

# 13 Underperformer Diagnostic
with tabs[12]:
    st.markdown("### Underperformer Diagnostic")
    st.info("This tab compares each store against expected productivity from its state, size band, maturity cohort, and region. It avoids labeling every low-volume store as bad without context.")
    diag = scope.copy()
    diag["Diagnostic Severity"] = rank_pct(-diag.get("Rev/SqFt Gap %", pd.Series(np.nan, index=diag.index)), True).fillna(50)
    c1,c2=st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(diag.sort_values("Diagnostic Severity", ascending=False), "Diagnostic Severity", "Store Number - Name", "Largest Underperformance Gaps", "h"), use_container_width=True, key="plotly_chart_1440")
    with c2:
        fig=px.scatter(diag, x="Peer Expected Rev/SqFt", y="Revenue per Sq. Ft.", color="Store Diagnostic", size="24 Volume Dollars", hover_name="Store Number - Name", title="Actual vs Peer-Expected Revenue / Sq. Ft.")
        fig.add_trace(go.Scatter(x=[diag["Peer Expected Rev/SqFt"].min(), diag["Peer Expected Rev/SqFt"].max()], y=[diag["Peer Expected Rev/SqFt"].min(), diag["Peer Expected Rev/SqFt"].max()], mode="lines", name="Expected line"))
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10), xaxis_tickprefix="$", yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1445")
    cols=safe_col_list(diag,["Store Number - Name","State","Size Band","Maturity Band","24 Volume Dollars","Revenue per Sq. Ft.","Peer Expected Rev/SqFt","Rev/SqFt Gap %","Store Diagnostic","State Risk_Score"])
    display_df(diag.sort_values("Diagnostic Severity", ascending=False)[cols], money_cols=["24 Volume Dollars","Revenue per Sq. Ft.","Peer Expected Rev/SqFt"], pct_cols=["Rev/SqFt Gap %"], height=430)

# 14 Growth Projection
with tabs[13]:
    st.markdown("### Macro-Adjusted Growth Projection")
    st.info("This model is explainable: it blends measured store momentum, state momentum, GDP segment signals, population growth, saturation, maturity, and uncertainty bands. It is not a black-box forecast.")
    model_col1, model_col2, model_col3 = st.columns(3)
    with model_col1:
        projection_horizon = st.slider("Projection horizon", 2, 8, 6, 1, key="growth_projection_horizon")
    with model_col2:
        scenario = st.selectbox("Scenario", ["Conservative", "Base", "Optimistic"], index=1, key="growth_projection_scenario")
    with model_col3:
        focus_level = st.selectbox("Projection focus", ["Selected view", "All states", "All stores"], index=0, key="growth_projection_focus")
    target_year = 2024 + projection_horizon
    growth_states, growth_projection, growth_stores, coef_df = build_growth_projection_model(stores, summary, pop, gdp, horizon_years=projection_horizon, scenario=scenario)
    if selected_state != "ALL" and focus_level == "Selected view":
        state_model_view = growth_states[growth_states["State Abbr"].eq(selected_state)].copy()
        state_projection_view = growth_projection[growth_projection["State Abbr"].eq(selected_state)].copy()
        store_model_view = growth_stores[growth_stores["State Abbr"].eq(selected_state)].copy()
    else:
        state_model_view = growth_states.copy(); state_projection_view = growth_projection.copy(); store_model_view = growth_stores.copy()
    st.markdown("#### Model structure")
    st.write("The projection uses five layers: store momentum, state trend, macro GDP/population trend, saturation/white-space pressure, and maturity/productivity adjustment. Low-store-count states are pulled toward the national trend to reduce one-store distortion.")
    if not state_model_view.empty:
        total_2024 = state_model_view["Volume_2024"].sum()
        total_target = state_model_view[f"Projected_{target_year}_Volume"].sum()
        target_change = total_target - total_2024
        blended_growth = (total_target / total_2024) ** (1 / projection_horizon) - 1 if total_2024 else np.nan
        k1,k2,k3,k4=st.columns(4)
        k1.metric(f"Projected {target_year} Volume", fmt_money(total_target), delta=fmt_money(target_change))
        k2.metric("Implied Annual Growth", fmt_pct(blended_growth))
        k3.metric("Avg Model Confidence", fmt_float(state_model_view["Model_Confidence"].mean(),1)+"/100")
        k4.metric("States in Model View", fmt_num(state_model_view["State Abbr"].nunique()))
    c1,c2=st.columns(2)
    with c1:
        if not state_projection_view.empty:
            national_proj = state_projection_view.groupby("Year", as_index=False).agg(**{"Projected Volume":("Projected Volume","sum"),"Low Case Volume":("Low Case Volume","sum"),"High Case Volume":("High Case Volume","sum")}) if (selected_state == "ALL" or focus_level != "Selected view") else state_projection_view
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=national_proj["Year"], y=national_proj["High Case Volume"], mode="lines", name="High", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=national_proj["Year"], y=national_proj["Low Case Volume"], mode="lines", name="Low", fill="tonexty", line=dict(width=0), fillcolor="rgba(128,128,128,.22)", showlegend=False))
            fig.add_trace(go.Scatter(x=national_proj["Year"], y=national_proj["Projected Volume"], mode="lines+markers", name="Base Projection"))
            fig.update_layout(title=f"Projected Volume Path — {scenario}", height=430, margin=dict(l=10,r=10,t=50,b=10), yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1489")
    with c2:
        rank_states = state_model_view.sort_values(f"Projected_{target_year}_Change", ascending=False)
        if not rank_states.empty:
            st.plotly_chart(bar_chart(rank_states, "State", f"Projected_{target_year}_Change", f"Projected Volume Change by {target_year}"), use_container_width=True, key="plotly_chart_1493")
    state_cols=safe_col_list(state_model_view,["State","State Abbr","Store_Count","Volume_2024",f"Projected_{target_year}_Volume",f"Projected_{target_year}_Change","Projected_Growth_Rate","Macro_Growth_Signal","Population_Growth_23_24","GDP_CAGR_10Y","CAGR_21_24","Growth_24_vs_23","Stores_per_1M_People","Revenue_per_Capita","Model_Confidence","Uncertainty"])
    st.markdown("#### State projection table")
    display_df(state_model_view[state_cols].sort_values(f"Projected_{target_year}_Change", ascending=False), money_cols=["Volume_2024",f"Projected_{target_year}_Volume",f"Projected_{target_year}_Change","Revenue_per_Capita"], pct_cols=["Projected_Growth_Rate","Macro_Growth_Signal","Population_Growth_23_24","GDP_CAGR_10Y","CAGR_21_24","Growth_24_vs_23","Uncertainty"], height=360)
    st.markdown("#### Store projection table")
    store_cols=safe_col_list(store_model_view,["Store Number - Name","City","State","24 Volume Dollars",f"Projected {target_year} Volume","Projected Volume Change","Projected Store Growth","Projection Uncertainty","Productivity Index","Store Diagnostic"])
    display_df(store_model_view[store_cols].sort_values("Projected Volume Change", ascending=False), money_cols=["24 Volume Dollars",f"Projected {target_year} Volume","Projected Volume Change"], pct_cols=["Projected Store Growth","Projection Uncertainty"], height=360)
    if not coef_df.empty:
        st.markdown("#### Model coefficients")
        display_df(coef_df, height=260)

# 15 Stability
with tabs[14]:
    st.markdown("### Volume Stability / Consistency")
    st.info("Stability helps separate reliable stores from stores with large year-to-year swings.")
    c1,c2=st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(scope.sort_values("Stability Score", ascending=False), "Stability Score", "Store Number - Name", "Most Stable Stores", "h"), use_container_width=True, key="plotly_chart_1510")
    with c2:
        st.plotly_chart(bar_chart(scope.sort_values("Stability Score", ascending=True), "Stability Score", "Store Number - Name", "Most Volatile Stores", "h"), use_container_width=True, key="plotly_chart_1512")
    cols=safe_col_list(scope,["Store Number - Name","State","21 Volume Dollars","22 Volume Dollars","23 Volume Dollars","24 Volume Dollars","Volume CV","Stability Score","Productivity Index"])
    display_df(scope.sort_values("Stability Score", ascending=False)[cols], money_cols=["21 Volume Dollars","22 Volume Dollars","23 Volume Dollars","24 Volume Dollars"], height=430)
