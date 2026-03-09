import os
import json
import base64
from io import BytesIO
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

TREASURY_SERIES_MAP = {
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
}

SHORT_RATE_SERIES_MAP = {
    "Prime": "DPRIME",
    "Fed Target Upper": "DFEDTARU",
    "SOFR": "SOFR",
}

UNEMPLOYMENT_SERIES_ID = "UNRATE"
CPI_SERIES_ID = "CPIAUCSL"
PAYROLL_SERIES_ID = "PAYEMS"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_recipients(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def fetch_fred_series(series_id: str, api_key: str, start_date: str, end_date: str) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
    }

    response = requests.get(FRED_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    observations = data.get("observations", [])
    if not observations:
        raise RuntimeError(f"No observations returned for series {series_id}")

    rows = []
    for obs in observations:
        value = obs.get("value")
        if value == ".":
            continue
        rows.append(
            {
                "date": pd.to_datetime(obs["date"]),
                "value": float(value),
            }
        )

    if not rows:
        raise RuntimeError(f"No valid numeric observations found for series {series_id}")

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df["value"]


def build_frame(series_map: dict[str, str], api_key: str, lookback_days: int) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days)

    series_frames = []
    for label, fred_id in series_map.items():
        s = fetch_fred_series(
            series_id=fred_id,
            api_key=api_key,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        s.name = label
        series_frames.append(s)

    return pd.concat(series_frames, axis=1).sort_index()


def last_complete_row_on_or_before(df: pd.DataFrame, target_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series]:
    subset = df.loc[:target_date].dropna(how="any")
    if subset.empty:
        raise RuntimeError(f"No complete data found on or before {target_date.date()}")
    return subset.index[-1], subset.iloc[-1]


def last_value_on_or_before_series(series: pd.Series, target_date: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    subset = series.loc[:target_date].dropna()
    if subset.empty:
        raise RuntimeError(f"No data found on or before {target_date.date()}")
    return subset.index[-1], float(subset.iloc[-1])


def format_rate(value: float) -> str:
    return f"{value:.2f}%"


def format_bps(value: float) -> str:
    return f"{value:+.0f} bps"


def format_pp(value: float) -> str:
    return f"{value:+.1f} pp"


def format_thousands(value: float) -> str:
    return f"{value:,.0f}k"


def period_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


def build_treasury_metrics(df: pd.DataFrame) -> dict:
    complete = df.dropna(how="any")
    if complete.empty:
        raise RuntimeError("No dates found where all treasury yields are available.")

    latest_date = complete.index[-1]
    current = complete.iloc[-1]

    week_target = latest_date - pd.Timedelta(days=7)
    year_target = latest_date - pd.DateOffset(years=1)

    week_date, week_row = last_complete_row_on_or_before(complete, week_target)
    year_date, year_row = last_complete_row_on_or_before(complete, year_target)

    metrics = {
        "latest_date": latest_date,
        "week_date": week_date,
        "year_date": year_date,
        "rates": {},
    }

    for tenor in ["2Y", "5Y", "10Y"]:
        current_value = float(current[tenor])
        week_value = float(week_row[tenor])
        year_value = float(year_row[tenor])

        metrics["rates"][tenor] = {
            "current": current_value,
            "week_ago": week_value,
            "year_ago": year_value,
            "wow_bps": (current_value - week_value) * 100.0,
            "yoy_bps": (current_value - year_value) * 100.0,
        }

    current_spread = metrics["rates"]["10Y"]["current"] - metrics["rates"]["2Y"]["current"]
    week_spread = metrics["rates"]["10Y"]["week_ago"] - metrics["rates"]["2Y"]["week_ago"]
    year_spread = metrics["rates"]["10Y"]["year_ago"] - metrics["rates"]["2Y"]["year_ago"]

    metrics["spread"] = {
        "current": current_spread,
        "week_ago": week_spread,
        "year_ago": year_spread,
        "wow_bps": (current_spread - week_spread) * 100.0,
        "yoy_bps": (current_spread - year_spread) * 100.0,
    }

    return metrics


def build_short_rate_metrics(df: pd.DataFrame) -> dict:
    complete = df.dropna(how="any")
    if complete.empty:
        raise RuntimeError("No dates found where all short-rate series are available.")

    latest_date = complete.index[-1]
    current = complete.iloc[-1]

    week_target = latest_date - pd.Timedelta(days=7)
    year_target = latest_date - pd.DateOffset(years=1)

    week_date, week_row = last_complete_row_on_or_before(complete, week_target)
    year_date, year_row = last_complete_row_on_or_before(complete, year_target)

    metrics = {
        "latest_date": latest_date,
        "week_date": week_date,
        "year_date": year_date,
        "rates": {},
    }

    for label in ["Prime", "Fed Target Upper", "SOFR"]:
        current_value = float(current[label])
        week_value = float(week_row[label])
        year_value = float(year_row[label])

        metrics["rates"][label] = {
            "current": current_value,
            "week_ago": week_value,
            "year_ago": year_value,
            "wow_bps": (current_value - week_value) * 100.0,
            "yoy_bps": (current_value - year_value) * 100.0,
        }

    return metrics


def build_unemployment_metrics(api_key: str, lookback_days: int = 450) -> dict:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days)

    series = fetch_fred_series(
        series_id=UNEMPLOYMENT_SERIES_ID,
        api_key=api_key,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    ).sort_index()

    latest_date, current_value = last_value_on_or_before_series(series, series.index[-1])
    year_target = latest_date - pd.DateOffset(years=1)
    year_date, year_value = last_value_on_or_before_series(series, year_target)

    return {
        "series": series,
        "latest_date": latest_date,
        "year_date": year_date,
        "current": current_value,
        "year_ago": year_value,
        "yoy_pp": current_value - year_value,
    }


def build_monthly_macro_metrics(api_key: str, lookback_days: int = 3650) -> dict:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days)

    cpi = fetch_fred_series(
        series_id=CPI_SERIES_ID,
        api_key=api_key,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    ).sort_index()

    payroll = fetch_fred_series(
        series_id=PAYROLL_SERIES_ID,
        api_key=api_key,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    ).sort_index()

    cpi_yoy = cpi.pct_change(12) * 100.0
    payroll_change = payroll.diff()

    cpi_yoy_clean = cpi_yoy.dropna()
    payroll_change_clean = payroll_change.dropna()

    latest_cpi_date, latest_cpi_yoy = last_value_on_or_before_series(
        cpi_yoy_clean,
        cpi_yoy_clean.index[-1],
    )
    prev_cpi_date = cpi_yoy_clean.index[-2]
    prev_cpi_value = float(cpi_yoy_clean.loc[prev_cpi_date])

    latest_payroll_date, latest_payroll_change = last_value_on_or_before_series(
        payroll_change_clean,
        payroll_change_clean.index[-1],
    )

    prior_12m_window = payroll_change_clean.iloc[-13:-1]
    if len(prior_12m_window) < 12:
        raise RuntimeError("Not enough payroll history to calculate prior 12-month average.")

    prior_12m_avg_payroll_change = float(prior_12m_window.mean())

    return {
        "cpi_level_series": cpi,
        "cpi_yoy_series": cpi_yoy_clean,
        "payroll_level_series": payroll,
        "payroll_change_series": payroll_change_clean,
        "latest_cpi_date": latest_cpi_date,
        "latest_cpi_yoy": latest_cpi_yoy,
        "prev_cpi_date": prev_cpi_date,
        "prev_cpi_yoy": prev_cpi_value,
        "latest_payroll_date": latest_payroll_date,
        "latest_payroll_change": latest_payroll_change,
        "prior_12m_start": prior_12m_window.index[0],
        "prior_12m_end": prior_12m_window.index[-1],
        "prior_12m_avg_payroll_change": prior_12m_avg_payroll_change,
        "payroll_vs_12m_avg": latest_payroll_change - prior_12m_avg_payroll_change,
    }


def load_state(state_file: str) -> dict:
    path = Path(state_file)
    if not path.exists():
        return {
            "last_sent_cpi_period": "",
            "last_sent_payroll_period": "",
        }

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "last_sent_cpi_period": "",
            "last_sent_payroll_period": "",
        }


def save_state(state: dict, state_file: str) -> None:
    Path(state_file).write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_include_monthly_macro(monthly_metrics: dict, state: dict) -> bool:
    latest_cpi_period = period_key(monthly_metrics["latest_cpi_date"])
    latest_payroll_period = period_key(monthly_metrics["latest_payroll_date"])

    last_sent_cpi_period = state.get("last_sent_cpi_period", "")
    last_sent_payroll_period = state.get("last_sent_payroll_period", "")


    return (
        latest_cpi_period != last_sent_cpi_period
        and latest_payroll_period != last_sent_payroll_period
    )


def update_state_after_monthly_send(monthly_metrics: dict, state: dict, state_file: str) -> None:
    state["last_sent_cpi_period"] = period_key(monthly_metrics["latest_cpi_date"])
    state["last_sent_payroll_period"] = period_key(monthly_metrics["latest_payroll_date"])
    save_state(state, state_file)


def make_commentary(treasury_metrics: dict, short_rate_metrics: dict) -> str:
    spread_change = treasury_metrics["spread"]["wow_bps"]

    if spread_change > 1:
        curve_text = "The Treasury curve steepened over the last week"
    elif spread_change < -1:
        curve_text = "The Treasury curve flattened over the last week"
    else:
        curve_text = "The Treasury curve was little changed over the last week"

    weekly_moves = {
        tenor: treasury_metrics["rates"][tenor]["wow_bps"] for tenor in ["2Y", "5Y", "10Y"]
    }
    biggest_tenor = max(weekly_moves, key=lambda x: abs(weekly_moves[x]))
    biggest_move = weekly_moves[biggest_tenor]

    if biggest_move > 0:
        move_text = f"with the {biggest_tenor} rising the most"
    elif biggest_move < 0:
        move_text = f"with the {biggest_tenor} falling the most"
    else:
        move_text = "with minimal movement across the tracked Treasury tenors"

    sofr_move = short_rate_metrics["rates"]["SOFR"]["wow_bps"]
    if sofr_move > 5:
        funding_text = "Short-term funding conditions also moved higher."
    elif sofr_move < -5:
        funding_text = "Short-term funding conditions eased over the week."
    else:
        funding_text = "Short-term funding conditions were relatively stable."

    current_spread = treasury_metrics["spread"]["current"] * 100.0
    return f"{curve_text}, {move_text}. The 10Y–2Y spread is now {current_spread:.0f} bps. {funding_text}"


def build_treasury_chart(df: pd.DataFrame, latest_date: pd.Timestamp, chart_days: int) -> bytes:
    chart_start = latest_date - pd.Timedelta(days=chart_days)
    chart_df = df.loc[chart_start:latest_date, ["2Y", "5Y", "10Y"]].dropna(how="all")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for tenor in ["2Y", "5Y", "10Y"]:
        ax.plot(chart_df.index, chart_df[tenor], label=tenor, linewidth=2)

    ax.set_title("U.S. Treasury Yields — Last 12 Months")
    ax.set_ylabel("Yield (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def build_unemployment_chart(series: pd.Series, latest_date: pd.Timestamp, chart_days: int) -> bytes:
    chart_start = latest_date - pd.Timedelta(days=chart_days)
    chart_series = series.loc[chart_start:latest_date].dropna()

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(chart_series.index, chart_series.values, linewidth=2)

    ax.set_title("U.S. Unemployment Rate — Last 12 Months")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def build_inflation_chart(cpi_yoy_series: pd.Series, years: int = 5) -> bytes:
    chart_start = cpi_yoy_series.index.max() - pd.DateOffset(years=years)
    chart_series = cpi_yoy_series.loc[chart_start:].dropna()

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(chart_series.index, chart_series.values, linewidth=2)

    ax.set_title("Inflation — Headline CPI YoY")
    ax.set_ylabel("YoY %")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def build_payroll_chart(payroll_change_series: pd.Series, months: int = 24) -> bytes:
    chart_series = payroll_change_series.tail(months).copy()

    positions = list(range(len(chart_series)))
    labels = [dt.strftime("%b %y") for dt in chart_series.index]

    if len(chart_series) >= 13:
        prior_12m_avg = float(chart_series.iloc[-13:-1].mean())
    else:
        prior_12m_avg = float(chart_series.mean())

    fig, ax = plt.subplots(figsize=(8.0, 3.8))

    ax.bar(positions, chart_series.values, width=0.72, label="Monthly payroll change")
    ax.axhline(0, linewidth=1)
    ax.axhline(prior_12m_avg, linestyle="--", linewidth=1.5, label="Prior 12M avg")

    tick_idx = list(range(0, len(positions), 2))
    if tick_idx[-1] != len(positions) - 1:
        tick_idx.append(len(positions) - 1)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right")

    ax.set_title("Job Growth — Monthly Change in Nonfarm Payrolls")
    ax.set_ylabel("Change (Thousands)")
    ax.set_xlabel("Month")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def build_email_bodies(
    treasury_metrics: dict,
    short_rate_metrics: dict,
    unemployment_metrics: dict,
    monthly_metrics: dict,
    include_monthly_macro: bool,
    commentary: str,
) -> tuple[str, str, str]:
    latest_treasury_date = treasury_metrics["latest_date"].strftime("%B %d, %Y")
    week_treasury_date = treasury_metrics["week_date"].strftime("%B %d, %Y")
    year_treasury_date = treasury_metrics["year_date"].strftime("%B %d, %Y")

    unemp_latest = unemployment_metrics["latest_date"].strftime("%B %Y")
    unemp_year = unemployment_metrics["year_date"].strftime("%B %Y")

    subject = f"Treasury & Economic Metrics Feed — {latest_treasury_date}"

    r2 = treasury_metrics["rates"]["2Y"]
    r5 = treasury_metrics["rates"]["5Y"]
    r10 = treasury_metrics["rates"]["10Y"]
    spread = treasury_metrics["spread"]

    prime = short_rate_metrics["rates"]["Prime"]
    fed = short_rate_metrics["rates"]["Fed Target Upper"]
    sofr = short_rate_metrics["rates"]["SOFR"]

    u_now = unemployment_metrics["current"]
    u_year = unemployment_metrics["year_ago"]
    u_yoy = unemployment_metrics["yoy_pp"]

    monthly_html = ""
    monthly_text = ""

    if include_monthly_macro:
        latest_cpi_month = monthly_metrics["latest_cpi_date"].strftime("%B %Y")
        prev_cpi_month = monthly_metrics["prev_cpi_date"].strftime("%B %Y")
        latest_payroll_month = monthly_metrics["latest_payroll_date"].strftime("%B %Y")
        prior_12m_start = monthly_metrics["prior_12m_start"].strftime("%b %Y")
        prior_12m_end = monthly_metrics["prior_12m_end"].strftime("%b %Y")

        monthly_html = f"""
        <h3 style="margin-top: 26px;">Monthly Macro Update</h3>

        <table cellpadding="8" cellspacing="0" border="1" style="border-collapse: collapse; font-size: 14px; margin-bottom: 18px;">
          <tr style="background-color: #f3f3f3;">
            <th align="left">Metric</th>
            <th align="right">Latest</th>
            <th align="right">Benchmark</th>
            <th align="right">Delta</th>
          </tr>
          <tr>
            <td>Headline CPI YoY</td>
            <td align="right">{monthly_metrics["latest_cpi_yoy"]:.2f}% ({latest_cpi_month})</td>
            <td align="right">{monthly_metrics["prev_cpi_yoy"]:.2f}% ({prev_cpi_month})</td>
            <td align="right">{monthly_metrics["latest_cpi_yoy"] - monthly_metrics["prev_cpi_yoy"]:+.2f} pp</td>
          </tr>
          <tr>
            <td>Monthly Payroll Growth</td>
            <td align="right">{format_thousands(monthly_metrics["latest_payroll_change"])} ({latest_payroll_month})</td>
            <td align="right">{format_thousands(monthly_metrics["prior_12m_avg_payroll_change"])} ({prior_12m_start}–{prior_12m_end} avg)</td>
            <td align="right">{format_thousands(monthly_metrics["payroll_vs_12m_avg"])}</td>
          </tr>
        </table>

        <img src="cid:inflation_chart" alt="Inflation Chart" style="width: 620px; max-width: 100%; border: 1px solid #ddd; margin-bottom: 18px;" />
        <br>
        <img src="cid:payroll_chart" alt="Payroll Chart" style="width: 620px; max-width: 100%; border: 1px solid #ddd;" />
        """

        monthly_text = f"""

Monthly Macro Update:
Headline CPI YoY: {monthly_metrics["latest_cpi_yoy"]:.2f}% ({latest_cpi_month}) | Prior: {monthly_metrics["prev_cpi_yoy"]:.2f}% ({prev_cpi_month})
Monthly Payroll Growth: {format_thousands(monthly_metrics["latest_payroll_change"])} ({latest_payroll_month}) | Prior 12M Avg: {format_thousands(monthly_metrics["prior_12m_avg_payroll_change"])} ({prior_12m_start}–{prior_12m_end}) | Delta: {format_thousands(monthly_metrics["payroll_vs_12m_avg"])}
""".rstrip()

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #222;">
        <h2 style="margin-bottom: 8px;">Treasury & Economic Metrics Feed</h2>
        <p style="margin-top: 0; color: #555;">
          Latest Treasury data date: <strong>{latest_treasury_date}</strong><br>
          Week-ago Treasury comparison date: <strong>{week_treasury_date}</strong><br>
          Year-ago Treasury comparison date: <strong>{year_treasury_date}</strong>
        </p>

        <p><strong>Market note:</strong> {commentary}</p>

        <h3 style="margin-top: 18px;">Treasury Snapshot</h3>
        <table cellpadding="8" cellspacing="0" border="1" style="border-collapse: collapse; font-size: 14px;">
          <tr style="background-color: #f3f3f3;">
            <th align="left">Metric</th>
            <th align="right">Current</th>
            <th align="right">1 Week Ago</th>
            <th align="right">WoW</th>
            <th align="right">1 Year Ago</th>
            <th align="right">YoY</th>
          </tr>
          <tr>
            <td>2Y</td>
            <td align="right">{format_rate(r2["current"])}</td>
            <td align="right">{format_rate(r2["week_ago"])}</td>
            <td align="right">{format_bps(r2["wow_bps"])}</td>
            <td align="right">{format_rate(r2["year_ago"])}</td>
            <td align="right">{format_bps(r2["yoy_bps"])}</td>
          </tr>
          <tr>
            <td>5Y</td>
            <td align="right">{format_rate(r5["current"])}</td>
            <td align="right">{format_rate(r5["week_ago"])}</td>
            <td align="right">{format_bps(r5["wow_bps"])}</td>
            <td align="right">{format_rate(r5["year_ago"])}</td>
            <td align="right">{format_bps(r5["yoy_bps"])}</td>
          </tr>
          <tr>
            <td>10Y</td>
            <td align="right">{format_rate(r10["current"])}</td>
            <td align="right">{format_rate(r10["week_ago"])}</td>
            <td align="right">{format_bps(r10["wow_bps"])}</td>
            <td align="right">{format_rate(r10["year_ago"])}</td>
            <td align="right">{format_bps(r10["yoy_bps"])}</td>
          </tr>
          <tr>
            <td>10Y–2Y Spread</td>
            <td align="right">{format_bps(spread["current"] * 100)}</td>
            <td align="right">{format_bps(spread["week_ago"] * 100)}</td>
            <td align="right">{format_bps(spread["wow_bps"])}</td>
            <td align="right">{format_bps(spread["year_ago"] * 100)}</td>
            <td align="right">{format_bps(spread["yoy_bps"])}</td>
          </tr>
        </table>

        <h3 style="margin-top: 22px;">Treasury Yield Chart</h3>
        <img src="cid:treasury_chart" alt="Treasury Yield Chart" style="width: 700px; max-width: 100%; border: 1px solid #ddd;" />

        <h3 style="margin-top: 26px;">Short-Rate Snapshot</h3>
        <table cellpadding="8" cellspacing="0" border="1" style="border-collapse: collapse; font-size: 14px;">
          <tr style="background-color: #f3f3f3;">
            <th align="left">Metric</th>
            <th align="right">Current</th>
            <th align="right">1 Week Ago</th>
            <th align="right">WoW</th>
            <th align="right">1 Year Ago</th>
            <th align="right">YoY</th>
          </tr>
          <tr>
            <td>Prime</td>
            <td align="right">{format_rate(prime["current"])}</td>
            <td align="right">{format_rate(prime["week_ago"])}</td>
            <td align="right">{format_bps(prime["wow_bps"])}</td>
            <td align="right">{format_rate(prime["year_ago"])}</td>
            <td align="right">{format_bps(prime["yoy_bps"])}</td>
          </tr>
          <tr>
            <td>Fed Target Upper</td>
            <td align="right">{format_rate(fed["current"])}</td>
            <td align="right">{format_rate(fed["week_ago"])}</td>
            <td align="right">{format_bps(fed["wow_bps"])}</td>
            <td align="right">{format_rate(fed["year_ago"])}</td>
            <td align="right">{format_bps(fed["yoy_bps"])}</td>
          </tr>
          <tr>
            <td>SOFR</td>
            <td align="right">{format_rate(sofr["current"])}</td>
            <td align="right">{format_rate(sofr["week_ago"])}</td>
            <td align="right">{format_bps(sofr["wow_bps"])}</td>
            <td align="right">{format_rate(sofr["year_ago"])}</td>
            <td align="right">{format_bps(sofr["yoy_bps"])}</td>
          </tr>
        </table>

        <h3 style="margin-top: 26px;">Labor Market Snapshot</h3>
        <table cellpadding="8" cellspacing="0" border="1" style="border-collapse: collapse; font-size: 14px;">
          <tr style="background-color: #f3f3f3;">
            <th align="left">Metric</th>
            <th align="right">Now</th>
            <th align="right">1 Year Ago</th>
            <th align="right">Change</th>
          </tr>
          <tr>
            <td>Unemployment Rate</td>
            <td align="right">{format_rate(u_now)}</td>
            <td align="right">{format_rate(u_year)} ({unemp_year})</td>
            <td align="right">{format_pp(u_yoy)}</td>
          </tr>
        </table>

        <p style="color: #555; margin-top: 8px;">
          Latest unemployment reading date: <strong>{unemp_latest}</strong>
        </p>

        <img src="cid:unemployment_chart" alt="Unemployment Chart" style="width: 620px; max-width: 100%; border: 1px solid #ddd;" />

        {monthly_html}

        <p style="margin-top: 24px; color: #666; font-size: 12px;">
          Automated weekly treasury and macro feed.
        </p>
      </body>
    </html>
    """

    text = f"""
Treasury & Economic Metrics Feed
Latest Treasury data date: {latest_treasury_date}
Week-ago Treasury comparison date: {week_treasury_date}
Year-ago Treasury comparison date: {year_treasury_date}

Market note:
{commentary}

Treasury Snapshot:
2Y: {format_rate(r2["current"])} | WoW: {format_bps(r2["wow_bps"])} | YoY: {format_bps(r2["yoy_bps"])}
5Y: {format_rate(r5["current"])} | WoW: {format_bps(r5["wow_bps"])} | YoY: {format_bps(r5["yoy_bps"])}
10Y: {format_rate(r10["current"])} | WoW: {format_bps(r10["wow_bps"])} | YoY: {format_bps(r10["yoy_bps"])}
10Y-2Y Spread: {format_bps(spread["current"] * 100)} | WoW: {format_bps(spread["wow_bps"])} | YoY: {format_bps(spread["yoy_bps"])}

Short-Rate Snapshot:
Prime: {format_rate(prime["current"])} | WoW: {format_bps(prime["wow_bps"])} | YoY: {format_bps(prime["yoy_bps"])}
Fed Target Upper: {format_rate(fed["current"])} | WoW: {format_bps(fed["wow_bps"])} | YoY: {format_bps(fed["yoy_bps"])}
SOFR: {format_rate(sofr["current"])} | WoW: {format_bps(sofr["wow_bps"])} | YoY: {format_bps(sofr["yoy_bps"])}

Labor Market Snapshot:
Unemployment Rate: {format_rate(u_now)}
1 Year Ago ({unemp_year}): {format_rate(u_year)}
Change: {format_pp(u_yoy)}
{monthly_text}

Charts are embedded in the HTML version.
""".strip()

    return subject, html, text


def get_gmail_service(credentials_file: str, token_file: str):
    creds = None
    token_path = Path(token_file)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def create_message(
    sender: str,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
    images: dict[str, bytes],
) -> dict:
    root = MIMEMultipart("related")
    root["To"] = ", ".join(recipients)
    root["From"] = sender
    root["Subject"] = subject

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text_body, "plain"))
    alt.attach(MIMEText(html_body, "html"))
    root.attach(alt)

    for cid, image_bytes in images.items():
        image = MIMEImage(image_bytes, _subtype="png")
        image.add_header("Content-ID", f"<{cid}>")
        image.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        root.attach(image)

    raw_message = base64.urlsafe_b64encode(root.as_bytes()).decode("utf-8")
    return {"raw": raw_message}


def send_email(
    service,
    sender: str,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
    images: dict[str, bytes],
) -> None:
    message = create_message(
        sender=sender,
        recipients=recipients,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        images=images,
    )
    service.users().messages().send(userId="me", body=message).execute()


def main():
    load_dotenv()

    fred_api_key = require_env("FRED_API_KEY")
    sender_email = require_env("SENDER_EMAIL")
    recipients = parse_recipients(require_env("RECIPIENTS"))

    lookback_days = int(os.getenv("LOOKBACK_DAYS", "450"))
    chart_days = int(os.getenv("CHART_DAYS", "365"))
    credentials_file = os.getenv("CREDENTIALS_FILE", "credentials.json")
    token_file = os.getenv("TOKEN_FILE", "token.json")
    subject_prefix = os.getenv("EMAIL_SUBJECT_PREFIX", "").strip()
    state_file = os.getenv("STATE_FILE", "report_state.json")

    if not Path(credentials_file).exists():
        raise RuntimeError(
            f"Missing {credentials_file}. Download your Gmail OAuth desktop credentials "
            f"from Google Cloud and place the file next to this script."
        )

    treasury_df = build_frame(TREASURY_SERIES_MAP, fred_api_key, lookback_days)
    short_rate_df = build_frame(SHORT_RATE_SERIES_MAP, fred_api_key, lookback_days)

    treasury_metrics = build_treasury_metrics(treasury_df)
    short_rate_metrics = build_short_rate_metrics(short_rate_df)
    unemployment_metrics = build_unemployment_metrics(api_key=fred_api_key, lookback_days=lookback_days)
    monthly_metrics = build_monthly_macro_metrics(api_key=fred_api_key, lookback_days=3650)

    state = load_state(state_file)
    include_monthly_macro = should_include_monthly_macro(monthly_metrics, state)

    commentary = make_commentary(treasury_metrics, short_rate_metrics)
    subject, html_body, text_body = build_email_bodies(
        treasury_metrics=treasury_metrics,
        short_rate_metrics=short_rate_metrics,
        unemployment_metrics=unemployment_metrics,
        monthly_metrics=monthly_metrics,
        include_monthly_macro=include_monthly_macro,
        commentary=commentary,
    )

    if subject_prefix:
        subject = f"{subject_prefix} {subject}"

    images = {
        "treasury_chart": build_treasury_chart(
            treasury_df,
            treasury_metrics["latest_date"],
            chart_days,
        ),
        "unemployment_chart": build_unemployment_chart(
            unemployment_metrics["series"],
            unemployment_metrics["latest_date"],
            chart_days,
        ),
    }

    if include_monthly_macro:
        images["inflation_chart"] = build_inflation_chart(monthly_metrics["cpi_yoy_series"], years=5)
        images["payroll_chart"] = build_payroll_chart(monthly_metrics["payroll_change_series"], months=24)

    service = get_gmail_service(credentials_file=credentials_file, token_file=token_file)

    send_email(
        service=service,
        sender=sender_email,
        recipients=recipients,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        images=images,
    )

    if include_monthly_macro:
        update_state_after_monthly_send(monthly_metrics, state, state_file)
        print(f"Email sent. Monthly macro block included. State updated in {state_file}.")
    else:
        print("Email sent. Monthly macro block not included this week.")


if __name__ == "__main__":
    main()
