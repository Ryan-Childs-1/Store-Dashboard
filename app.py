# app.py
# ============================================================
# Store AI Dashboard
# GitHub/Streamlit-ready dashboard using:
#   1) bundled US population data
#   2) bundled BEA GDP-by-state data
#   3) user-uploaded Store List v1.csv
#
# Creator-ready, single-file Streamlit build.
# ============================================================

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# -----------------------------
# App constants
# -----------------------------
APP_NAME = "Store AI Dashboard"
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

# Approximate state centroids. Exact address-level plotting is used automatically if lat/lon columns exist.
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
YEAR_MAP = {"21 Volume": 2021, "22 Volume": 2022, "23 Volume": 2023, "24 Volume": 2024}

# -----------------------------
# Page styling
# -----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🗺️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015));
        box-shadow: 0 6px 18px rgba(0,0,0,.05);
    }
    .small-note {font-size: .84rem; color: #7a7a7a;}
    .section-title {font-size:1.2rem; font-weight:700; margin-top:.8rem; margin-bottom:.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utility helpers
# -----------------------------
def clean_state_name(x: object) -> str:
    if pd.isna(x):
        return ""
    text = str(x).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def yes_like(x: object) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"yes", "done", "y", "true", "1", "x", "new"}


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def money_fmt(x: float, unit: str = "") -> str:
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:,.2f}M{unit}"
    if abs(x) >= 1_000:
        return f"{x/1_000:,.1f}K{unit}"
    return f"{x:,.2f}{unit}"


def pct_fmt(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:,.1f}%"


def deterministic_jitter(n: int, seed_key: str, scale_lat: float = 1.4, scale_lon: float = 1.8) -> np.ndarray:
    seed = abs(hash(seed_key)) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, max(n, 1), endpoint=False)
    rng.shuffle(angles)
    radii = np.sqrt(rng.uniform(0.05, 1.0, max(n, 1)))
    return np.c_[np.sin(angles) * radii * scale_lat, np.cos(angles) * radii * scale_lon][:n]

# -----------------------------
# Lightweight from-scratch LSTM-reservoir model
# -----------------------------
@dataclass
class ForecastResult:
    history: pd.DataFrame
    forecast: pd.DataFrame
    quality: Dict[str, float]
    method: str


class ScratchLSTMReservoir:
    """Tiny LSTM-reservoir forecaster implemented with NumPy gate equations.

    It avoids deep-learning dependencies by using fixed recurrent LSTM weights and training
    a ridge readout on the hidden state. This gives a fast, reproducible AI-style sequence
    model suitable for Streamlit Cloud and small state-level economic/store time series.
    """

    def __init__(self, hidden_size: int = 24, seed: int = 11, alpha: float = 1.0):
        self.hidden_size = hidden_size
        self.seed = seed
        self.alpha = alpha
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.readout = Ridge(alpha=alpha)
        self.params = {}
        self.is_fit = False

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def _init_params(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        h = self.hidden_size
        def w(shape, scale=0.35):
            return rng.normal(0, scale, shape)
        self.params = {
            "Wf": w((input_dim, h)), "Uf": w((h, h), 0.12), "bf": np.ones(h) * 0.8,
            "Wi": w((input_dim, h)), "Ui": w((h, h), 0.12), "bi": np.zeros(h),
            "Wo": w((input_dim, h)), "Uo": w((h, h), 0.12), "bo": np.zeros(h),
            "Wc": w((input_dim, h)), "Uc": w((h, h), 0.12), "bc": np.zeros(h),
        }

    def _encode_sequence(self, X_seq: np.ndarray) -> np.ndarray:
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        p = self.params
        for x in X_seq:
            f = self._sigmoid(x @ p["Wf"] + h @ p["Uf"] + p["bf"])
            i = self._sigmoid(x @ p["Wi"] + h @ p["Ui"] + p["bi"])
            o = self._sigmoid(x @ p["Wo"] + h @ p["Uo"] + p["bo"])
            g = np.tanh(x @ p["Wc"] + h @ p["Uc"] + p["bc"])
            c = f * c + i * g
            h = o * np.tanh(c)
        return h

    def fit_predict(self, years: Iterable[int], values: Iterable[float], horizon: int = 4, window: int = 4) -> ForecastResult:
        raw = pd.DataFrame({"Year": list(years), "Actual": list(values)}).dropna().sort_values("Year")
        raw["Actual"] = pd.to_numeric(raw["Actual"], errors="coerce")
        raw = raw.dropna()
        if len(raw) < 4 or raw["Actual"].nunique() <= 1:
            return naive_forecast(raw, horizon=horizon, method="Naive fallback: insufficient sequence length")

        vals = raw["Actual"].astype(float).to_numpy()
        yrs = raw["Year"].astype(float).to_numpy()
        window = int(min(max(2, window), max(2, len(raw) - 1)))

        # Features include normalized value, one-period pct change, and normalized time.
        pct = np.r_[0.0, np.diff(vals) / np.where(vals[:-1] == 0, 1.0, np.abs(vals[:-1]))]
        time_idx = (yrs - yrs.min()) / max(1.0, yrs.max() - yrs.min())
        feat = np.c_[vals, pct, time_idx]
        feat_s = self.scaler_x.fit_transform(feat)
        y_s = self.scaler_y.fit_transform(vals.reshape(-1, 1)).ravel()
        self._init_params(input_dim=feat_s.shape[1])

        X_states, y_targets = [], []
        for end in range(window, len(feat_s)):
            X_states.append(self._encode_sequence(feat_s[end-window:end]))
            y_targets.append(y_s[end])

        if len(X_states) < 2:
            return naive_forecast(raw, horizon=horizon, method="Naive fallback: very short target sequence")

        X_states = np.vstack(X_states)
        y_targets = np.array(y_targets)
        self.readout.fit(X_states, y_targets)
        self.is_fit = True

        fitted_scaled = self.readout.predict(X_states)
        fitted = self.scaler_y.inverse_transform(fitted_scaled.reshape(-1, 1)).ravel()
        actual = vals[window:]
        mae = float(np.mean(np.abs(fitted - actual))) if len(actual) else np.nan
        mape = float(np.mean(np.abs((fitted - actual) / np.where(actual == 0, 1, actual)))) if len(actual) else np.nan

        # Recursive forecast.
        future_rows = []
        values_ext = vals.astype(float).tolist()
        years_ext = yrs.astype(int).tolist()
        for step in range(1, horizon + 1):
            next_year = int(years_ext[-1] + 1)
            last_vals = np.array(values_ext, dtype=float)
            next_pct = 0.0 if len(last_vals) < 2 or last_vals[-2] == 0 else (last_vals[-1] - last_vals[-2]) / abs(last_vals[-2])
            future_time = (next_year - yrs.min()) / max(1.0, yrs.max() - yrs.min())
            last_feat_rows = []
            for i in range(max(0, len(values_ext) - window), len(values_ext)):
                vi = values_ext[i]
                pi = 0.0 if i == 0 or values_ext[i-1] == 0 else (values_ext[i] - values_ext[i-1]) / abs(values_ext[i-1])
                ti = (years_ext[i] - yrs.min()) / max(1.0, yrs.max() - yrs.min())
                last_feat_rows.append([vi, pi, ti])
            # Pad if needed.
            while len(last_feat_rows) < window:
                last_feat_rows.insert(0, last_feat_rows[0])
            seq_s = self.scaler_x.transform(np.array(last_feat_rows[-window:]))
            h_state = self._encode_sequence(seq_s)
            pred_s = self.readout.predict(h_state.reshape(1, -1))[0]
            pred = float(self.scaler_y.inverse_transform([[pred_s]])[0, 0])
            # Stabilize against tiny data over-extrapolation.
            recent = np.array(values_ext[-min(4, len(values_ext)):], dtype=float)
            recent_growth = np.nanmedian(np.diff(recent) / np.where(recent[:-1] == 0, 1, np.abs(recent[:-1]))) if len(recent) >= 2 else 0
            cap_low = values_ext[-1] * (1 + max(-0.45, recent_growth - 0.25))
            cap_high = values_ext[-1] * (1 + min(0.45, recent_growth + 0.25))
            if values_ext[-1] >= 0:
                pred = float(np.clip(pred, max(0.0, cap_low), max(cap_low, cap_high)))
            future_rows.append({"Year": next_year, "Forecast": pred})
            values_ext.append(pred)
            years_ext.append(next_year)

        hist = raw.copy()
        return ForecastResult(
            history=hist,
            forecast=pd.DataFrame(future_rows),
            quality={"MAE": mae, "MAPE": mape, "Training Points": float(len(X_states))},
            method="Scratch LSTM-reservoir + ridge readout",
        )


def naive_forecast(raw: pd.DataFrame, horizon: int = 4, method: str = "Naive fallback") -> ForecastResult:
    if raw.empty:
        return ForecastResult(pd.DataFrame(columns=["Year", "Actual"]), pd.DataFrame(columns=["Year", "Forecast"]), {}, method)
    vals = raw["Actual"].astype(float).to_numpy()
    yrs = raw["Year"].astype(int).to_numpy()
    if len(vals) >= 2 and vals[-2] != 0:
        g = np.nanmedian(np.diff(vals[-min(4, len(vals)):]) / np.where(vals[-min(4, len(vals)):-1] == 0, 1, np.abs(vals[-min(4, len(vals)):-1])))
        g = float(np.clip(g, -0.25, 0.25))
    else:
        g = 0.0
    future = []
    last = float(vals[-1])
    y = int(yrs[-1])
    for _ in range(horizon):
        y += 1
        last = max(0.0, last * (1 + g))
        future.append({"Year": y, "Forecast": last})
    return ForecastResult(raw, pd.DataFrame(future), {"MAE": np.nan, "MAPE": np.nan, "Training Points": 0.0}, method)

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_population() -> pd.DataFrame:
    pop = pd.read_csv(POP_FILE)
    pop["State"] = pop["US State"].map(clean_state_name)
    pop["State Abbr"] = pop["State"].map(STATE_ABBR)
    for c in ["Population 2024", "Population 2023", "Growth Rate", "% of US", "Density (/mile2)"]:
        if c in pop.columns:
            pop[c] = safe_num(pop[c])
    return pop


@st.cache_data(show_spinner=False)
def load_gdp() -> pd.DataFrame:
    gdp = pd.read_csv(GDP_FILE)
    gdp["GeoName"] = gdp["GeoName"].astype(str).str.replace("*", "", regex=False).str.strip()
    years = [c for c in gdp.columns if re.fullmatch(r"\d{4}", str(c))]
    for c in years:
        gdp[c] = safe_num(gdp[c])
    gdp["State"] = gdp["GeoName"].map(clean_state_name)
    gdp["State Abbr"] = gdp["State"].map(STATE_ABBR)
    return gdp


def read_store_upload(uploaded_file) -> pd.DataFrame:
    data = uploaded_file.getvalue()
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"Could not read uploaded store list. Last error: {last_err}")


def normalize_store_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "State" not in df.columns:
        raise ValueError("The store list must include a 'State' column.")
    df["State"] = df["State"].map(clean_state_name)
    df["State Abbr"] = df["State"].map(STATE_ABBR)
    # Try abbreviation fallback if the State field is already an abbreviation.
    missing = df["State Abbr"].isna()
    df.loc[missing, "State Abbr"] = df.loc[missing, "State"].str.upper().map(lambda x: x if x in ABBR_STATE else np.nan)
    df.loc[df["State"].str.len() == 2, "State"] = df.loc[df["State"].str.len() == 2, "State"].str.upper().map(ABBR_STATE).fillna(df["State"])

    for c in ["Sq. Footage", "Grand Opening"] + VOLUME_COLUMNS:
        if c in df.columns:
            df[c] = safe_num(df[c])
    for c in FLAG_COLUMNS:
        if c in df.columns:
            df[c + " Flag"] = df[c].map(yes_like)

    # Coordinates: exact if present, otherwise state centroid plus deterministic jitter.
    lat_candidates = [c for c in df.columns if c.lower() in {"lat", "latitude", "store_latitude"}]
    lon_candidates = [c for c in df.columns if c.lower() in {"lon", "lng", "long", "longitude", "store_longitude"}]
    if lat_candidates and lon_candidates:
        df["Latitude"] = safe_num(df[lat_candidates[0]])
        df["Longitude"] = safe_num(df[lon_candidates[0]])
        df["Coordinate Source"] = "Uploaded lat/lon"
    else:
        df["Latitude"] = np.nan
        df["Longitude"] = np.nan
        df["Coordinate Source"] = "State centroid + deterministic jitter"
        for abbr, idx in df.groupby("State Abbr").groups.items():
            if abbr in STATE_CENTROIDS:
                lat, lon = STATE_CENTROIDS[abbr]
                indices = list(idx)
                jit = deterministic_jitter(len(indices), str(abbr))
                df.loc[indices, "Latitude"] = lat + jit[:, 0]
                df.loc[indices, "Longitude"] = lon + jit[:, 1]
    return df

# -----------------------------
# Aggregation and forecasting
# -----------------------------
def build_state_summary(stores: pd.DataFrame, pop: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
    if stores.empty:
        base = pd.DataFrame({"State Abbr": list(ABBR_STATE.keys())})
    else:
        agg = stores.groupby(["State Abbr", "State"], dropna=False).agg(
            Store_Count=("Store Number - Name", "count") if "Store Number - Name" in stores.columns else ("State", "count"),
            Sq_Footage=("Sq. Footage", "sum") if "Sq. Footage" in stores.columns else ("State", "size"),
            Avg_Sq_Footage=("Sq. Footage", "mean") if "Sq. Footage" in stores.columns else ("State", "size"),
            Volume_2024=("24 Volume", "sum") if "24 Volume" in stores.columns else ("State", "size"),
            Volume_2023=("23 Volume", "sum") if "23 Volume" in stores.columns else ("State", "size"),
            Volume_2022=("22 Volume", "sum") if "22 Volume" in stores.columns else ("State", "size"),
            Volume_2021=("21 Volume", "sum") if "21 Volume" in stores.columns else ("State", "size"),
        ).reset_index()
        base = agg
    pop_small = pop[["State", "State Abbr", "Population 2024", "Growth Rate", "Density (/mile2)"]].dropna(subset=["State Abbr"])
    base = base.merge(pop_small, on="State Abbr", how="outer", suffixes=("", "_pop"))
    base["State"] = base["State"].fillna(base.get("State_pop")).fillna(base["State Abbr"].map(ABBR_STATE))

    gdp_total = gdp[gdp["Description"].astype(str).str.strip().eq("All industry total")].copy()
    if "2020" in gdp_total.columns:
        gdp_small = gdp_total[["State Abbr", "2020"]].dropna(subset=["State Abbr"]).rename(columns={"2020": "GDP_2020_Millions"})
        base = base.merge(gdp_small, on="State Abbr", how="left")
    for c in ["Store_Count", "Sq_Footage", "Avg_Sq_Footage", "Volume_2024", "Volume_2023", "Volume_2022", "Volume_2021"]:
        if c in base.columns:
            base[c] = base[c].fillna(0)
    base["Volume_per_Store_2024"] = base["Volume_2024"] / base["Store_Count"].replace(0, np.nan)
    base["Volume_per_SqFt_2024"] = base["Volume_2024"] / base["Sq_Footage"].replace(0, np.nan)
    base["Stores_per_Million_People"] = base["Store_Count"] / (base["Population 2024"] / 1_000_000).replace(0, np.nan)
    base["Volume_per_Capita"] = base["Volume_2024"] / base["Population 2024"].replace(0, np.nan)
    return base


def volume_series_for_scope(stores: pd.DataFrame, state_abbr: Optional[str]) -> pd.DataFrame:
    df = stores if state_abbr in {None, "ALL"} else stores[stores["State Abbr"] == state_abbr]
    rows = []
    for col, year in YEAR_MAP.items():
        if col in df.columns:
            rows.append({"Year": year, "Actual": float(df[col].sum(skipna=True))})
    return pd.DataFrame(rows).sort_values("Year")


def gdp_series_for_scope(gdp: pd.DataFrame, state_abbr: Optional[str]) -> pd.DataFrame:
    years = [c for c in gdp.columns if re.fullmatch(r"\d{4}", str(c))]
    total = gdp[gdp["Description"].astype(str).str.strip().eq("All industry total")].copy()
    if state_abbr in {None, "ALL"}:
        row = total[total["GeoName"].eq("United States")]
    else:
        row = total[total["State Abbr"].eq(state_abbr)]
    if row.empty:
        return pd.DataFrame(columns=["Year", "Actual"])
    r = row.iloc[0]
    return pd.DataFrame({"Year": [int(y) for y in years], "Actual": [float(r[y]) for y in years]}).dropna()


@st.cache_data(show_spinner=False)
def cached_forecast(years_tuple: Tuple[int, ...], vals_tuple: Tuple[float, ...], horizon: int, seed: int, window: int) -> ForecastResult:
    model = ScratchLSTMReservoir(hidden_size=28, seed=seed, alpha=0.8)
    return model.fit_predict(years_tuple, vals_tuple, horizon=horizon, window=window)


def forecast_series(series_df: pd.DataFrame, horizon: int = 4, seed: int = 11, window: int = 4) -> ForecastResult:
    if series_df.empty:
        return ForecastResult(series_df, pd.DataFrame(columns=["Year", "Forecast"]), {}, "No data")
    return cached_forecast(tuple(series_df["Year"].astype(int)), tuple(series_df["Actual"].astype(float)), horizon, seed, window)


def plot_history_forecast(result: ForecastResult, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    if not result.history.empty:
        fig.add_trace(go.Scatter(x=result.history["Year"], y=result.history["Actual"], mode="lines+markers", name="Actual"))
    if not result.forecast.empty:
        fig.add_trace(go.Scatter(x=result.forecast["Year"], y=result.forecast["Forecast"], mode="lines+markers", name="AI forecast", line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title="Year", yaxis_title=y_label, height=360, margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h"))
    return fig


def flag_summary(scope_stores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(scope_stores)
    for c in FLAG_COLUMNS:
        fc = c + " Flag"
        if fc in scope_stores.columns:
            count = int(scope_stores[fc].sum())
            rows.append({"Metric": c, "Stores": count, "Share": count / n if n else 0})
    return pd.DataFrame(rows).sort_values(["Stores", "Metric"], ascending=[False, True]) if rows else pd.DataFrame()


def make_store_map(scope_stores: pd.DataFrame, selected_state: str) -> go.Figure:
    if scope_stores.empty:
        fig = go.Figure()
        fig.update_layout(height=450, title="No stores to map")
        return fig
    hover_cols = [c for c in ["Store Number - Name", "City", "State", "Address", "24 Volume", "Sq. Footage", "Volume Band"] if c in scope_stores.columns]
    fig = px.scatter_geo(
        scope_stores,
        lat="Latitude",
        lon="Longitude",
        scope="usa",
        hover_name="Store Number - Name" if "Store Number - Name" in scope_stores.columns else None,
        hover_data=hover_cols,
        size="24 Volume" if "24 Volume" in scope_stores.columns else None,
        color="State Abbr" if selected_state == "ALL" else "Volume Band" if "Volume Band" in scope_stores.columns else None,
        projection="albers usa",
        title="Store Locations" + (" — United States" if selected_state == "ALL" else f" — {ABBR_STATE.get(selected_state, selected_state)}"),
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h"))
    return fig


def make_us_choropleth(summary: pd.DataFrame, metric: str) -> go.Figure:
    metric = metric if metric in summary.columns else "Store_Count"
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
            "Population 2024": ":,.0f" if "Population 2024" in summary.columns else False,
            metric: ":,.2f",
        },
        title=f"US State Map — {metric.replace('_', ' ')}",
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def ai_estimate_table(scope_stores: pd.DataFrame, pop_row: pd.Series, volume_fc: ForecastResult, gdp_fc: ForecastResult, selected_label: str) -> pd.DataFrame:
    current_volume = float(scope_stores["24 Volume"].sum()) if "24 Volume" in scope_stores.columns and not scope_stores.empty else 0.0
    store_count = len(scope_stores)
    sq_ft = float(scope_stores["Sq. Footage"].sum()) if "Sq. Footage" in scope_stores.columns and not scope_stores.empty else np.nan
    avg_store = current_volume / store_count if store_count else np.nan
    next_vol = float(volume_fc.forecast.iloc[0]["Forecast"]) if not volume_fc.forecast.empty else np.nan
    next_gdp = float(gdp_fc.forecast.iloc[0]["Forecast"]) if not gdp_fc.forecast.empty else np.nan
    pop = float(pop_row.get("Population 2024", np.nan)) if isinstance(pop_row, pd.Series) else np.nan
    growth = float(pop_row.get("Growth Rate", np.nan)) if isinstance(pop_row, pd.Series) else np.nan
    pop_next = pop * (1 + growth) if not pd.isna(pop) and not pd.isna(growth) else np.nan
    projected_stores = store_count
    if store_count and not pd.isna(pop_next) and pop:
        demand_pressure = (next_vol / current_volume - 1) if current_volume else 0
        pop_pressure = (pop_next / pop - 1) if pop else 0
        projected_stores = max(store_count, store_count * (1 + 0.35 * max(0, demand_pressure) + 0.15 * max(0, pop_pressure)))

    rows = [
        {"Metric": "Current Stores", "Current": store_count, "AI Next-Year Estimate": projected_stores, "Notes": "Store count pressure estimate from volume and population trend."},
        {"Metric": "Total Volume", "Current": current_volume, "AI Next-Year Estimate": next_vol, "Notes": volume_fc.method},
        {"Metric": "Average Volume / Store", "Current": avg_store, "AI Next-Year Estimate": next_vol / projected_stores if projected_stores else np.nan, "Notes": "Forecast volume divided by estimated store base."},
        {"Metric": "Total Sq. Footage", "Current": sq_ft, "AI Next-Year Estimate": sq_ft * (projected_stores / store_count) if store_count and not pd.isna(sq_ft) else sq_ft, "Notes": "Scaled by projected store base."},
        {"Metric": "Volume / Sq. Foot", "Current": current_volume / sq_ft if sq_ft else np.nan, "AI Next-Year Estimate": next_vol / (sq_ft * (projected_stores / store_count)) if store_count and sq_ft else np.nan, "Notes": "Efficiency estimate."},
        {"Metric": "Population", "Current": pop, "AI Next-Year Estimate": pop_next, "Notes": "Uses supplied 2024 population growth rate."},
        {"Metric": "GDP, All Industry Total", "Current": float(gdp_fc.history.iloc[-1]["Actual"]) if not gdp_fc.history.empty else np.nan, "AI Next-Year Estimate": next_gdp, "Notes": gdp_fc.method},
    ]
    out = pd.DataFrame(rows)
    out.insert(0, "Scope", selected_label)
    return out

# -----------------------------
# UI
# -----------------------------
st.title("🗺️ Store AI Dashboard")
st.caption("Interactive US state dashboard with uploaded store data, bundled population/GDP data, and from-scratch LSTM-reservoir forecasts.")

with st.sidebar:
    st.header("1) Upload Store List")
    uploaded_store = st.file_uploader("Upload `Store List v1.csv`", type=["csv"])
    st.caption("The repo bundles the population and GDP files. The store list is intentionally uploaded at runtime.")
    st.divider()
    st.header("2) Dashboard Controls")
    forecast_horizon = st.slider("Forecast horizon", 1, 8, 4, 1)
    choropleth_metric = st.selectbox(
        "US map color metric",
        ["Store_Count", "Volume_2024", "Volume_per_Store_2024", "Stores_per_Million_People", "Volume_per_Capita", "GDP_2020_Millions", "Population 2024"],
        index=1,
    )
    st.divider()
    st.markdown("**Coordinate note**")
    st.caption("If the uploaded file has latitude/longitude columns, exact points are used. Otherwise, stores are placed around state centroids with deterministic jitter.")

pop = load_population()
gdp = load_gdp()

if uploaded_store is None:
    st.info("Upload `Store List v1.csv` in the sidebar to activate the dashboard.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Bundled Population Data")
        st.dataframe(pop.head(10), use_container_width=True)
    with c2:
        st.subheader("Bundled GDP Data")
        st.dataframe(gdp.head(10), use_container_width=True)
    st.stop()

try:
    raw_stores = read_store_upload(uploaded_store)
    stores = normalize_store_df(raw_stores)
except Exception as exc:
    st.error(f"Could not load the uploaded store list: {exc}")
    st.stop()

summary = build_state_summary(stores, pop, gdp)
valid_states = sorted(stores.dropna(subset=["State Abbr"])["State Abbr"].unique().tolist(), key=lambda x: ABBR_STATE.get(x, x))
state_options = ["ALL"] + valid_states

# Plotly click selection support varies by Streamlit version and chart type, so a selectbox is provided as a reliable control.
selected_state = st.selectbox(
    "Click into a state / choose analysis scope",
    state_options,
    format_func=lambda x: "United States / National View" if x == "ALL" else f"{ABBR_STATE.get(x, x)} ({x})",
)
selected_label = "United States" if selected_state == "ALL" else ABBR_STATE.get(selected_state, selected_state)

scope_stores = stores if selected_state == "ALL" else stores[stores["State Abbr"] == selected_state]
state_summary_row = summary[summary["State Abbr"].eq(selected_state)].iloc[0] if selected_state != "ALL" and not summary[summary["State Abbr"].eq(selected_state)].empty else pd.Series(dtype="object")
if selected_state == "ALL":
    pop_row = pd.Series({
        "Population 2024": pop["Population 2024"].sum(),
        "Growth Rate": np.average(pop["Growth Rate"].fillna(0), weights=pop["Population 2024"].fillna(1)),
        "Density (/mile2)": np.nan,
    })
else:
    pr = pop[pop["State Abbr"].eq(selected_state)]
    pop_row = pr.iloc[0] if not pr.empty else pd.Series(dtype="object")

vol_series = volume_series_for_scope(stores, selected_state)
gdp_series = gdp_series_for_scope(gdp, selected_state)
vol_fc = forecast_series(vol_series, horizon=forecast_horizon, seed=23, window=2)
gdp_fc = forecast_series(gdp_series, horizon=forecast_horizon, seed=41, window=5)

# Top metrics
m1, m2, m3, m4, m5 = st.columns(5)
store_count = len(scope_stores)
vol24 = float(scope_stores["24 Volume"].sum()) if "24 Volume" in scope_stores.columns else np.nan
vol23 = float(scope_stores["23 Volume"].sum()) if "23 Volume" in scope_stores.columns else np.nan
sqft = float(scope_stores["Sq. Footage"].sum()) if "Sq. Footage" in scope_stores.columns else np.nan
next_vol = float(vol_fc.forecast.iloc[0]["Forecast"]) if not vol_fc.forecast.empty else np.nan
next_gdp = float(gdp_fc.forecast.iloc[0]["Forecast"]) if not gdp_fc.forecast.empty else np.nan
vol_growth = (vol24 / vol23 - 1) if vol23 else np.nan
m1.metric("Scope", selected_label)
m2.metric("Stores", f"{store_count:,.0f}")
m3.metric("2024 Volume", money_fmt(vol24), delta=pct_fmt(vol_growth) if not pd.isna(vol_growth) else None)
m4.metric("Sq. Footage", money_fmt(sqft))
m5.metric("AI Next-Year Volume", money_fmt(next_vol))

map_tab, state_tab, ai_tab, data_tab = st.tabs(["🗺️ Map Explorer", "📊 State/National Statistics", "🤖 AI Forecasts", "📁 Data Quality"])

with map_tab:
    left, right = st.columns([1.05, 1])
    with left:
        st.plotly_chart(make_us_choropleth(summary, choropleth_metric), use_container_width=True)
        st.caption("Use the state selector above to drill into a state. The map colors all states from the available store/population/GDP rollup.")
    with right:
        st.plotly_chart(make_store_map(scope_stores, selected_state), use_container_width=True)
        if not scope_stores.empty and "Coordinate Source" in scope_stores.columns:
            st.caption(f"Coordinate source: {scope_stores['Coordinate Source'].iloc[0]}")

with state_tab:
    st.subheader(f"Statistics — {selected_label}")
    c1, c2 = st.columns(2)
    with c1:
        state_metrics = {
            "Store count": store_count,
            "2021 volume": float(scope_stores["21 Volume"].sum()) if "21 Volume" in scope_stores.columns else np.nan,
            "2022 volume": float(scope_stores["22 Volume"].sum()) if "22 Volume" in scope_stores.columns else np.nan,
            "2023 volume": vol23,
            "2024 volume": vol24,
            "Average 2024 volume/store": vol24 / store_count if store_count else np.nan,
            "Total square footage": sqft,
            "Volume per square foot": vol24 / sqft if sqft else np.nan,
            "Population 2024": float(pop_row.get("Population 2024", np.nan)) if isinstance(pop_row, pd.Series) else np.nan,
            "Population growth rate": float(pop_row.get("Growth Rate", np.nan)) if isinstance(pop_row, pd.Series) else np.nan,
            "GDP latest in file, 2020 millions": float(gdp_series.iloc[-1]["Actual"]) if not gdp_series.empty else np.nan,
        }
        metric_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in state_metrics.items()])
        st.dataframe(metric_df, use_container_width=True)
    with c2:
        fs = flag_summary(scope_stores)
        st.markdown("**Operational / market flags**")
        if fs.empty:
            st.write("No flag columns found.")
        else:
            fig = px.bar(fs.head(15), x="Stores", y="Metric", orientation="h", text="Share", title="Top Store Flags")
            fig.update_traces(texttemplate="%{text:.0%}")
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Stores in selected scope**")
    display_cols = [c for c in ["Store Number - Name", "St. No.", "City", "Address", "State", "Sq. Footage", "Region", "24 Volume", "23 Volume", "Volume Band"] if c in scope_stores.columns]
    st.dataframe(scope_stores[display_cols].sort_values("24 Volume", ascending=False) if "24 Volume" in display_cols else scope_stores[display_cols], use_container_width=True, height=360)

with ai_tab:
    st.subheader(f"AI Forecasts — {selected_label}")
    st.caption("Models use a lightweight LSTM-reservoir built from scratch with NumPy gate equations and a trained ridge readout. It is designed for short state-level sequences and reliable Streamlit deployment.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_history_forecast(vol_fc, "Store Volume Forecast", "Volume"), use_container_width=True)
        st.caption(f"Method: {vol_fc.method} | MAPE: {pct_fmt(vol_fc.quality.get('MAPE', np.nan))}")
    with c2:
        st.plotly_chart(plot_history_forecast(gdp_fc, "GDP Forecast — All Industry Total", "Millions of current dollars"), use_container_width=True)
        st.caption(f"Method: {gdp_fc.method} | MAPE: {pct_fmt(gdp_fc.quality.get('MAPE', np.nan))}")

    st.markdown("### AI Estimate Matrix")
    ai_table = ai_estimate_table(scope_stores, pop_row, vol_fc, gdp_fc, selected_label)
    st.dataframe(ai_table, use_container_width=True)

    st.markdown("### Store-Level AI Scoring")
    scored = scope_stores.copy()
    if not scored.empty and "24 Volume" in scored.columns:
        vol_rank = scored["24 Volume"].rank(pct=True).fillna(0)
        sqft_eff = (scored["24 Volume"] / scored["Sq. Footage"].replace(0, np.nan)).rank(pct=True).fillna(0) if "Sq. Footage" in scored.columns else 0
        flag_cols = [c + " Flag" for c in FLAG_COLUMNS if c + " Flag" in scored.columns]
        flag_score = scored[flag_cols].mean(axis=1).fillna(0).rank(pct=True) if flag_cols else 0
        scored["AI Opportunity Score"] = (0.55 * vol_rank + 0.30 * sqft_eff + 0.15 * flag_score) * 100
        scored["AI Score Band"] = pd.cut(scored["AI Opportunity Score"], bins=[-1, 35, 65, 100], labels=["Stabilize", "Develop", "Lead"])
        score_cols = [c for c in ["Store Number - Name", "City", "State", "24 Volume", "Sq. Footage", "AI Opportunity Score", "AI Score Band"] if c in scored.columns]
        st.dataframe(scored[score_cols].sort_values("AI Opportunity Score", ascending=False), use_container_width=True, height=360)

with data_tab:
    st.subheader("Data Quality & Diagnostics")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Uploaded rows", f"{len(stores):,.0f}")
    d2.metric("Columns", f"{stores.shape[1]:,.0f}")
    d3.metric("States with stores", f"{stores['State Abbr'].nunique():,.0f}")
    d4.metric("Missing state rows", f"{stores['State Abbr'].isna().sum():,.0f}")

    with st.expander("Column audit", expanded=False):
        audit = pd.DataFrame({
            "Column": stores.columns,
            "Non-Null": [stores[c].notna().sum() for c in stores.columns],
            "Missing": [stores[c].isna().sum() for c in stores.columns],
            "Dtype": [str(stores[c].dtype) for c in stores.columns],
        })
        st.dataframe(audit, use_container_width=True)

    with st.expander("Raw uploaded store sample", expanded=False):
        st.dataframe(stores.head(50), use_container_width=True)

    with st.expander("State summary table", expanded=False):
        st.dataframe(summary.sort_values("Store_Count", ascending=False), use_container_width=True)

st.caption("Built for GitHub + Streamlit Cloud. Population/GDP files are bundled; the store list is uploaded at runtime.")
