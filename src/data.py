import io
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDUSTRIES_DAILY_ZIP = DATA_DIR / "49_Industry_Portfolios_daily_CSV.zip"
INDUSTRIES_MONTHLY_ZIP = DATA_DIR / "49_Industry_Portfolios_CSV.zip"
FACTORS_MONTHLY_ZIP = DATA_DIR / "F-F_Research_Data_Factors_CSV.zip"

START = "1970-01"
END = "2026-09"
EVAL_START = "1980-01"


def read_french_section(zip_path, marker):
    """Read the section of a French Data Library CSV whose header line contains `marker`.
    Returns decimal returns indexed by date (daily) or by month (monthly), missing values as NaN."""
    with zipfile.ZipFile(zip_path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()

    start = next(i for i, line in enumerate(lines) if marker in line)
    header = start if lines[start].startswith(",") else start + 1
    end = next(i for i in range(header + 1, len(lines)) if not lines[i].strip())

    df = pd.read_csv(io.StringIO("\n".join(lines[header:end])), index_col=0)
    dates = df.index.astype(str).str.strip()
    if (dates.str.len() == 8).all():
        df.index = pd.to_datetime(dates, format="%Y%m%d")
    else:
        df.index = pd.to_datetime(dates, format="%Y%m").to_period("M")
    df.columns = df.columns.str.strip()
    return df.mask(df <= -99.99) / 100


def month_ends(daily):
    """Last trading day of each month, indexed by month."""
    return daily.index.to_series().groupby(daily.index.to_period("M")).last()


def load_data(start=START, end=END, eval_start=EVAL_START):
    """Returns daily industry returns (covariance estimation), monthly industry returns,
    monthly total CRSP market return, monthly risk-free rate and rebalancing dates."""
    daily = read_french_section(INDUSTRIES_DAILY_ZIP, "Average Value Weighted Returns -- Daily")
    monthly = read_french_section(INDUSTRIES_MONTHLY_ZIP, "Average Value Weighted Returns -- Monthly")
    factors = read_french_section(FACTORS_MONTHLY_ZIP, ",Mkt-RF")

    daily = daily.loc[pd.Period(start, "M").start_time: pd.Period(end, "M").end_time]
    monthly = monthly.loc[start:end]
    factors = factors.loc[start:end]
    market = (factors["Mkt-RF"] + factors["RF"]).rename("Market")
    rf = factors["RF"].rename("RF")
    rebalance_dates = month_ends(daily)

    if not (rebalance_dates.index.equals(monthly.index) and monthly.index.equals(factors.index)):
        raise ValueError("Daily and monthly files do not cover the same months")

    eval_daily_start = pd.Period(eval_start, "M").start_time
    checks = [daily.loc[eval_daily_start:], monthly.loc[eval_start:], market.loc[eval_start:], rf.loc[eval_start:]]
    if any(df.isna().any(axis=None) for df in checks):
        raise ValueError(f"Missing data after {eval_start}")

    return daily, monthly, market, rf, rebalance_dates
