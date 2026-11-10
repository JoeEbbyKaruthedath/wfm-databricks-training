"""Helpers shared by the WFM training notebooks.

Builds the synthetic venues, staffing roles, calendar features and hourly guest demand
used in the demo. Everything here is synthetic. The notebooks import it with
`from wfm_utils import ...`, which works because this file sits next to them in the workspace.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Defaults for running the notebooks by hand. The pipelines get their catalog and schema from the bundle
# (databricks.yml), so change these only for interactive runs in your workspace.
DEFAULT_CATALOG = "classic_stable_catalog"
DEFAULT_SCHEMA = "wfm_training"

# Columns the demand model is trained on (same order everywhere).
FEATURES = [
    "base_demand",
    "venue_type_code",
    "park_code",
    "hour",
    "day_of_week",
    "month",
    "is_peak_season",
    "is_holiday",
    "is_evening_event",
]

VENUE_TYPE_CODES = {"quick_service": 0, "table_service": 1, "merchandise": 2, "attraction": 3}

# venue_type: (number of venues, base guests/hour range, open hour, close hour)
VENUE_TYPES = {
    "quick_service": (6, (150, 400), 9, 22),
    "table_service": (4, (60, 140), 11, 22),
    "merchandise": (5, (100, 300), 9, 22),
    "attraction": (5, (800, 1800), 9, 21),
}

# venue_type, role, guests one staff member can serve per hour, minimum staff on duty
ROLES = [
    ("quick_service", "cook", 40, 1),
    ("quick_service", "cashier", 60, 1),
    ("table_service", "cook", 30, 1),
    ("table_service", "server", 16, 1),
    ("table_service", "host", 80, 1),
    ("merchandise", "cashier", 70, 1),
    ("merchandise", "stocker", 150, 1),
    ("attraction", "operator", 250, 2),
    ("attraction", "greeter", 400, 1),
]

# Demand shape over the day: (peak hour, height, width) bumps on top of a base level.
_PROFILE_PEAKS = {
    "quick_service": [(12.5, 1.0, 1.2), (18.0, 0.8, 1.3)],
    "table_service": [(12.5, 0.7, 1.0), (19.0, 1.0, 1.2)],
    "merchandise": [(15.0, 0.7, 2.5), (20.5, 1.0, 1.0)],
    "attraction": [(13.0, 1.0, 3.0)],
}
_DOW_FACTOR = np.array([0.92, 0.90, 0.92, 0.97, 1.08, 1.25, 1.18])  # Mon..Sun
# Weekends lift attractions most and quick service least (exponent on the day-of-week factor).
_WEEKEND_SENSITIVITY = {"quick_service": 1.0, "table_service": 1.3, "merchandise": 1.2, "attraction": 1.6}
_MONTH_FACTOR = {1: 0.85, 2: 0.95, 3: 1.20, 4: 1.15, 5: 1.00, 6: 1.25,
                 7: 1.30, 8: 1.15, 9: 0.75, 10: 1.05, 11: 0.95, 12: 1.25}
_PEAK_MONTHS = [3, 4, 6, 7, 8, 12]
# Evening events (Fri/Sat nights, mid-September to October) pull big crowds after 6 PM.
_EVENT_EVENING_LIFT = {"quick_service": 1.6, "table_service": 1.5, "merchandise": 1.5, "attraction": 2.0}
_HOLIDAY_LIFT = {"quick_service": 1.25, "table_service": 1.3, "merchandise": 1.35, "attraction": 1.4}


def build_venues(seed: int = 7) -> pd.DataFrame:
    """20 synthetic venues across two parks."""
    rng = np.random.default_rng(seed)
    rows, n = [], 0
    labels = {"quick_service": "Quick Service", "table_service": "Table Service",
              "merchandise": "Merch Shop", "attraction": "Attraction"}
    for venue_type, (count, (lo, hi), open_hour, close_hour) in VENUE_TYPES.items():
        for i in range(count):
            n += 1
            park_code = n % 2
            rows.append({
                "venue_id": f"V{n:03d}",
                "venue_name": f"{labels[venue_type]} {i + 1:02d}",
                "park": f"Park {park_code + 1}",
                "park_code": park_code,
                "venue_type": venue_type,
                "venue_type_code": VENUE_TYPE_CODES[venue_type],
                "base_demand": float(rng.integers(lo, hi)),
                "open_hour": open_hour,
                "close_hour": close_hour,
            })
    return pd.DataFrame(rows)


def build_roles() -> pd.DataFrame:
    return pd.DataFrame(ROLES, columns=["venue_type", "role", "guests_per_staff_hour", "min_staff"])


def hourly_profile(venue_type: str, hours: np.ndarray) -> np.ndarray:
    level = 0.15 + sum(a * np.exp(-((hours - p) ** 2) / (2 * w ** 2)) for p, a, w in _PROFILE_PEAKS[venue_type])
    return level / level.max()


def calendar_features(dates) -> pd.DataFrame:
    d = pd.to_datetime(pd.Series(dates))
    month, day = d.dt.month, d.dt.day
    holiday = (
        d.dt.strftime("%m-%d").isin(["01-01", "07-04"])
        | ((month == 11) & day.between(24, 30))  # Thanksgiving week (approximate)
        | ((month == 12) & (day >= 20))          # winter holidays
    )
    event_night = d.dt.dayofweek.isin([4, 5]) & (((month == 9) & (day >= 15)) | (month == 10))
    return pd.DataFrame({
        "date": d.dt.date,
        "day_of_week": d.dt.dayofweek.astype("int64"),
        "month": month.astype("int64"),
        "is_peak_season": month.isin(_PEAK_MONTHS).astype("int64"),
        "is_holiday": holiday.astype("int64"),
        "is_evening_event": event_night.astype("int64"),
    })


def build_feature_frame(venues: pd.DataFrame, dates) -> pd.DataFrame:
    """One row per venue, date and open hour, with the model features."""
    cal = calendar_features(dates)
    frames = []
    for v in venues.itertuples(index=False):
        hours = pd.DataFrame({"hour": np.arange(v.open_hour, v.close_hour, dtype="int64")})
        f = cal.merge(hours, how="cross")
        f["venue_id"], f["park"], f["venue_type"] = v.venue_id, v.park, v.venue_type
        f["base_demand"] = float(v.base_demand)
        f["venue_type_code"] = int(v.venue_type_code)
        f["park_code"] = int(v.park_code)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out[["venue_type_code", "park_code"]] = out[["venue_type_code", "park_code"]].astype("int64")
    return out[["venue_id", "park", "venue_type", "date"] + FEATURES]


def generate_demand_history(venues: pd.DataFrame, days: int = 365, end: date | None = None,
                            seed: int = 11) -> pd.DataFrame:
    """Hourly guests per venue for the `days` days ending yesterday (or `end`)."""
    rng = np.random.default_rng(seed)
    end = end or (date.today() - timedelta(days=1))
    dates = [end - timedelta(days=i) for i in range(days - 1, -1, -1)]
    frame = build_feature_frame(venues, dates)

    # Patterns the model can learn from the features:
    vt = frame["venue_type"]
    hour = frame["hour"].to_numpy()
    month = frame["month"].to_numpy()
    profile = np.empty(len(frame))
    for t in vt.unique():
        m = (vt == t).to_numpy()
        profile[m] = hourly_profile(t, hour[m].astype(float))
    weekend = _DOW_FACTOR[frame["day_of_week"].to_numpy()] ** vt.map(_WEEKEND_SENSITIVITY).to_numpy()
    season = frame["month"].map(_MONTH_FACTOR).to_numpy()
    summer_evening = np.where(np.isin(month, [6, 7, 8]) & (hour >= 19), 1.25, 1.0)
    holiday = np.where(frame["is_holiday"].to_numpy() == 1, vt.map(_HOLIDAY_LIFT).to_numpy(), 1.0)
    event = np.where((frame["is_evening_event"].to_numpy() == 1) & (hour >= 18), vt.map(_EVENT_EVENING_LIFT).to_numpy(), 1.0)
    mean = frame["base_demand"].to_numpy() * profile * weekend * season * summer_evening * holiday * event

    # A little randomness the model can't predict: park-level day-to-day swings, and hour-to-hour noise.
    daily = {}
    for park in frame["park"].unique():
        daily.update({(park, d): x for d, x in zip(dates, rng.lognormal(0, 0.03, len(dates)))})
    day_noise = np.array([daily[(p, d)] for p, d in zip(frame["park"], frame["date"])])
    frame["guests"] = np.round(mean * day_noise * rng.lognormal(0, 0.05, len(frame))).astype("int64")
    return frame
