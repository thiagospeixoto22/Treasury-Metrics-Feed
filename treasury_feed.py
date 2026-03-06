import os
import base64
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
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
SERIES_MAP = {
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
}
UNEMPLOYMENT_SERIES_ID = "UNRATE"


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
        rows.append({
            "date": pd.to_datetime(obs["date"]),
            "value": float(value),
        })

    if not rows:
        raise RuntimeError(f"No valid numeric observations found for series {series_id}")

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df["value"]


def build_rates_frame(api_key: str, lookback_days: int) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days)

    series_frames = []
    for label, fred_id in SERIES_MAP.items():
        s = fetch_fred_series(
            series_id=fred_id,
            api_key=api_key,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        s.name = label
        series_frames.append(s)

    df = pd.concat(series_frames, axis=1).sort_index()
    return df


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


def format_pp(value: float) -> str:
    return f"{value:+.1f} pp"

def build_metrics(df: pd.DataFrame) -> dict:
    complete = df.dropna(how="any")
    if complete.empty:
        raise RuntimeError("No dates found where all 2Y, 5Y, and 10Y yields are available.")

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


def make_commentary(metrics: dict) -> str:
    spread_change = metrics["spread"]["wow_bps"]

    if spread_change > 1:
        curve_text = "The curve steepened over the last week"
    elif spread_change < -1:
        curve_text = "The curve flattened over the last week"
    else:
        curve_text = "The curve was little changed over the last week"

    weekly_moves = {
        tenor: metrics["rates"][tenor]["wow_bps"] for tenor in ["2Y", "5Y", "10Y"]
    }
    biggest_tenor = max(weekly_moves, key=lambda x: abs(weekly_moves[x]))
    biggest_move = weekly_moves[biggest_tenor]

    if biggest_move > 0:
        move_text = f"with the {biggest_tenor} rising the most"
    elif biggest_move < 0:
        move_text = f"with the {biggest_tenor} falling the most"
    else:
        move_text = "with minimal movement across the tracked tenors"

    current_spread = metrics["spread"]["current"] * 100.0
    return f"{curve_text}, {move_text}. The 10Y–2Y spread is now {current_spread:.0f} bps."


def format_rate(value: float) -> str:
    return f"{value:.2f}%"


def format_bps(value: float) -> str:
    return f"{value:+.0f} bps"


def build_chart(df: pd.DataFrame, latest_date: pd.Timestamp, chart_days: int) -> bytes:
    chart_start = latest_date - pd.Timedelta(days=chart_days)
    chart_df = df.loc[chart_start:latest_date, ["2Y", "5Y", "10Y"]].dropna(how="all")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))

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

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
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


def build_email_bodies(metrics: dict, unemployment_metrics: dict, commentary: str, report_tz: str) -> tuple[str, str, str]:
    tz = ZoneInfo(report_tz)

    latest_local = metrics["latest_date"].to_pydatetime().replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    week_local = metrics["week_date"].to_pydatetime().replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    year_local = metrics["year_date"].to_pydatetime().replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)

    unemp_latest_local = unemployment_metrics["latest_date"].to_pydatetime().replace(
        tzinfo=ZoneInfo("UTC")
    ).astimezone(tz)
    unemp_year_local = unemployment_metrics["year_date"].to_pydatetime().replace(
        tzinfo=ZoneInfo("UTC")
    ).astimezone(tz)

    subject = f"Treasury Metrics Feed — {latest_local.strftime('%B %d, %Y')}"

    r2 = metrics["rates"]["2Y"]
    r5 = metrics["rates"]["5Y"]
    r10 = metrics["rates"]["10Y"]
    spread = metrics["spread"]

    u_now = unemployment_metrics["current"]
    u_year = unemployment_metrics["year_ago"]
    u_yoy = unemployment_metrics["yoy_pp"]

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #222;">
        <h2 style="margin-bottom: 8px;">Treasury Metrics Feed</h2>
        <p style="margin-top: 0; color: #555;">
          Latest available Treasury data date: <strong>{latest_local.strftime('%B %d, %Y')}</strong><br>
          Week-ago Treasury comparison date: <strong>{week_local.strftime('%B %d, %Y')}</strong><br>
          Year-ago Treasury comparison date: <strong>{year_local.strftime('%B %d, %Y')}</strong>
        </p>

        <p><strong>Market note:</strong> {commentary}</p>

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

        <h3 style="margin-top: 26px;">Labor Market Snapshot</h3>
        <table cellpadding="8" cellspacing="0" border="1" style="border-collapse: collapse; font-size: 14px;">
          <tr style="background-color: #f3f3f3;">
            <th align="left">Metric</th>
            <th align="right">Now</th>
            <th align="right">Year-Ago Date</th>
            <th align="right">1 Year Ago</th>
            <th align="right">Change</th>
          </tr>
          <tr>
            <td>Unemployment Rate</td>
            <td align="right">{format_rate(u_now)}</td>
            <td align="right">{unemp_year_local.strftime('%B %Y')}</td>
            <td align="right">{format_rate(u_year)}</td>
            <td align="right">{format_pp(u_yoy)}</td>
          </tr>
        </table>

        <p style="color: #555; margin-top: 8px;">
          Latest unemployment reading date: <strong>{unemp_latest_local.strftime('%B %Y')}</strong>
        </p>

        <img src="cid:unemployment_chart" alt="Unemployment Chart" style="width: 620px; max-width: 100%; border: 1px solid #ddd;" />

        <p style="margin-top: 24px; color: #666; font-size: 12px;">
          Automated weekly treasury feed.
        </p>
      </body>
    </html>
    """

    text = f"""
Treasury Metrics Feed
Latest available Treasury data date: {latest_local.strftime('%B %d, %Y')}
Week-ago Treasury comparison date: {week_local.strftime('%B %d, %Y')}
Year-ago Treasury comparison date: {year_local.strftime('%B %d, %Y')}

Market note:
{commentary}

2Y: {format_rate(r2["current"])} | WoW: {format_bps(r2["wow_bps"])} | YoY: {format_bps(r2["yoy_bps"])}
5Y: {format_rate(r5["current"])} | WoW: {format_bps(r5["wow_bps"])} | YoY: {format_bps(r5["yoy_bps"])}
10Y: {format_rate(r10["current"])} | WoW: {format_bps(r10["wow_bps"])} | YoY: {format_bps(r10["yoy_bps"])}
10Y-2Y Spread: {format_bps(spread["current"] * 100)} | WoW: {format_bps(spread["wow_bps"])} | YoY: {format_bps(spread["yoy_bps"])}

Unemployment Rate:
Now: {format_rate(u_now)}
1 Year Ago ({unemp_year_local.strftime('%B %Y')}): {format_rate(u_year)}
Change: {format_pp(u_yoy)}

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

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service


def create_message(
    sender: str,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
    treasury_chart_bytes: bytes,
    unemployment_chart_bytes: bytes,
) -> dict:
    root = MIMEMultipart("related")
    root["To"] = ", ".join(recipients)
    root["From"] = sender
    root["Subject"] = subject

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text_body, "plain"))
    alt.attach(MIMEText(html_body, "html"))
    root.attach(alt)

    image1 = MIMEImage(treasury_chart_bytes, _subtype="png")
    image1.add_header("Content-ID", "<treasury_chart>")
    image1.add_header("Content-Disposition", "inline", filename="treasury_chart.png")
    root.attach(image1)

    image2 = MIMEImage(unemployment_chart_bytes, _subtype="png")
    image2.add_header("Content-ID", "<unemployment_chart>")
    image2.add_header("Content-Disposition", "inline", filename="unemployment_chart.png")
    root.attach(image2)

    raw_message = base64.urlsafe_b64encode(root.as_bytes()).decode("utf-8")
    return {"raw": raw_message}

def send_email(
    service,
    sender: str,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
    treasury_chart_bytes: bytes,
    unemployment_chart_bytes: bytes,
) -> None:
    message = create_message(
        sender=sender,
        recipients=recipients,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        treasury_chart_bytes=treasury_chart_bytes,
        unemployment_chart_bytes=unemployment_chart_bytes,
    )
    service.users().messages().send(userId="me", body=message).execute()

def main():
    load_dotenv()

    fred_api_key = require_env("FRED_API_KEY")
    sender_email = require_env("SENDER_EMAIL")
    recipients = parse_recipients(require_env("RECIPIENTS"))

    report_timezone = os.getenv("REPORT_TIMEZONE", "America/New_York")
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "450"))
    chart_days = int(os.getenv("CHART_DAYS", "365"))
    credentials_file = os.getenv("CREDENTIALS_FILE", "credentials.json")
    token_file = os.getenv("TOKEN_FILE", "token.json")
    subject_prefix = os.getenv("EMAIL_SUBJECT_PREFIX", "").strip()

    if not Path(credentials_file).exists():
        raise RuntimeError(
            f"Missing {credentials_file}. Download your Gmail OAuth desktop credentials "
            f"from Google Cloud and place the file next to this script."
        )

    rates_df = build_rates_frame(api_key=fred_api_key, lookback_days=lookback_days)
    metrics = build_metrics(rates_df)

    unemployment_metrics = build_unemployment_metrics(
        api_key=fred_api_key,
        lookback_days=lookback_days,
    )

    commentary = make_commentary(metrics)
    subject, html_body, text_body = build_email_bodies(
        metrics,
        unemployment_metrics,
        commentary,
        report_timezone,
    )

    if subject_prefix:
        subject = f"{subject_prefix} {subject}"

    treasury_chart_bytes = build_chart(rates_df, metrics["latest_date"], chart_days)
    unemployment_chart_bytes = build_unemployment_chart(
        unemployment_metrics["series"],
        unemployment_metrics["latest_date"],
        chart_days,
    )

    service = get_gmail_service(credentials_file=credentials_file, token_file=token_file)

    send_email(
        service=service,
        sender=sender_email,
        recipients=recipients,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        treasury_chart_bytes=treasury_chart_bytes,
        unemployment_chart_bytes=unemployment_chart_bytes,
    )

    print("Treasury feed email sent successfully.")


if __name__ == "__main__":
    main()
