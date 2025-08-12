import requests
import json
import pytz
import base64
import smtplib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# --- Import Credentials ---
from credentials import (
    KALSHI_API_KEY, KALSHI_RSA_PRIVATE_KEY, 
    SENDER_EMAIL, RECEIVER_EMAILS, EMAIL_APP_PASSWORD,
    BANKROLL_CACHE_PATH, PLACED_ORDERS_PATH
)

sender_email = SENDER_EMAIL
receiver_email = RECEIVER_EMAILS
app_password = EMAIL_APP_PASSWORD

# 🔑 Load RSA Key
private_key = serialization.load_pem_private_key(
    KALSHI_RSA_PRIVATE_KEY.encode(), password=None, backend=default_backend()
)

eastern = pytz.timezone("US/Eastern")

def sign_request(method, path, key_id, private_key):
    timestamp = str(int(datetime.now().timestamp() * 1000))
    message = timestamp + method + path.split('?')[0]
    signature = private_key.sign(
        message.encode('utf-8'),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode('utf-8'),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

def read_bankroll_cache():
    try:
        with open(BANKROLL_CACHE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_open_positions_from_yesterday():
    eastern = pytz.timezone("US/Eastern")
    today = datetime.now(eastern).date()
    yesterday = today - timedelta(days=1)
    eastern = pytz.timezone("US/Eastern")
    yesterday = datetime.now(eastern).date() - timedelta(days=1)
    start_ts = int(datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=eastern).timestamp())
    end_ts = int(datetime.combine(yesterday, datetime.max.time()).replace(tzinfo=eastern).timestamp())
    order_path = "/trade-api/v2/portfolio/orders"
    order_url = f"https://api.elections.kalshi.com{order_path}"
    order_headers = sign_request("GET", order_path, KALSHI_API_KEY, private_key)
    orders_resp = requests.get(order_url, headers=order_headers, params={"min_ts": start_ts, "max_ts": end_ts})
    orders = orders_resp.json().get("orders", [])

    bought_tickers = {o["ticker"] for o in orders if o["status"] == "executed" and o["action"] == "buy"}

    pos_path = "/trade-api/v2/portfolio/positions"
    pos_url = f"https://api.elections.kalshi.com{pos_path}"
    pos_headers = sign_request("GET", pos_path, KALSHI_API_KEY, private_key)
    positions = requests.get(pos_url, headers=pos_headers).json().get("positions", [])

    return [
        f"{pos['ticker']} — {pos['side']} {pos['count']} @ {pos['average_price']}¢"
        for pos in positions if pos["count"] > 0 and pos["ticker"] in bought_tickers
    ]

def summarize_sport(sport_prefix, sport_name, team_map):
    """Generic function to summarize bets for any sport"""
    from tabulate import tabulate
    import time

    def local_sign_request(method, path):
        ts = str(int(time.time() * 1000))
        msg = ts + method + path.split('?')[0]
        sig = private_key.sign(msg.encode(), padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ), hashes.SHA256())
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": KALSHI_API_KEY,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts
        }

    def get_orders():
        eastern = pytz.timezone("US/Eastern")
        yesterday = datetime.now(eastern).date() - timedelta(days=1)
        start_ts = int(datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=eastern).timestamp())
        end_ts = int(datetime.combine(yesterday, datetime.max.time()).replace(tzinfo=eastern).timestamp())
        path = "/trade-api/v2/portfolio/orders"
        r = requests.get(f"https://api.elections.kalshi.com{path}", headers=local_sign_request("GET", path), params={
            "min_ts": start_ts, "max_ts": end_ts
        })
        r.raise_for_status()
        return r.json().get("orders", [])

    def get_settlements():
        path = "/trade-api/v2/portfolio/settlements"
        r = requests.get(f"https://api.elections.kalshi.com{path}", headers=local_sign_request("GET", path))
        r.raise_for_status()
        return r.json().get("settlements", [])

    def get_available_markets():
        """Get count of available markets on Kalshi for this sport"""
        try:
            url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={sport_prefix}"
            headers = {"accept": "application/json"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return len([m for m in data.get("markets", []) if m.get("status") == "active"])
        except:
            return 0

    orders = [o for o in get_orders() if o.get("ticker", "").startswith(sport_prefix)]
    settlements = get_settlements()
    available_markets = get_available_markets()

    data = []
    total_wager_raw = 0.0
    total_return_raw = 0.0

    for o in orders:
        code = o["ticker"].split("-")[-1]
        team = team_map.get(code, code)
        side = o["side"]
        odds_val = (o["yes_price"] if side == "yes" else o["no_price"]) / 100
        odds = f"{int(round(odds_val * 100))}%"

        wager_raw = (o["taker_fill_cost"] + o["maker_fill_cost"]) / 100
        wager = f"${int(round(wager_raw))}"
        total_wager_raw += wager_raw

        settlement = next((s for s in settlements if s.get("ticker") == o["ticker"]), None)

        if settlement:
            market_result = settlement.get("market_result", "")
            revenue = settlement.get("revenue", 0) / 100
            return_amt = revenue
        else:
            return_amt = 0.0

        total_return_raw += return_amt

        if return_amt <= 0:
            return_str = "$0"
        else:
            return_str = f"${int(round(return_amt))}"

        try:
            with open(PLACED_ORDERS_PATH, "r") as f:
                placed_orders_data = json.load(f)
            placed_order = next((order for order in placed_orders_data if order["ticker"] == o["ticker"]), {})
            ev_after = f"{placed_order.get('expected_value_after_devig', 0):.1f}%" if placed_order else "N/A"
        except FileNotFoundError:
            ev_after = "N/A"

        data.append({
            "team": team,
            "odds": odds,
            "wager": wager,
            "return": return_str,
            "ev_after_devig": ev_after
        })

    data.append({
        "team": "TOTAL",
        "odds": "",
        "wager": f"${total_wager_raw:.2f}",
        "return": f"${total_return_raw:.2f}",
        "ev_after_devig": ""
    })

    df = pd.DataFrame(data, columns=["team", "odds", "wager", "return", "ev_after_devig"])
    
    games_bet = len([d for d in data if d["team"] != "TOTAL"])
    participation_rate = f"{games_bet}/{available_markets}" if available_markets > 0 else f"{games_bet}/0"
    
    return df, participation_rate

def summarize_mlb():
    """Backward compatibility wrapper for MLB"""
    TEAM_MAP = {
        "MLB_DET": "Detroit Tigers", "MLB_MIA": "Miami Marlins", "MLB_CWS": "Chicago White Sox",
        "MLB_LAA": "Los Angeles Angels", "MLB_LAD": "Los Angeles Dodgers", "MLB_SD": "San Diego Padres",
        "MLB_NYY": "New York Yankees", "MLB_TOR": "Toronto Blue Jays", "MLB_STL": "St. Louis Cardinals",
        "MLB_MIL": "Milwaukee Brewers", "MLB_ATL": "Atlanta Braves", "MLB_TEX": "Texas Rangers",
        "MLB_HOU": "Houston Astros", "MLB_BAL": "Baltimore Orioles", "MLB_PHI": "Philadelphia Phillies",
        "MLB_SEA": "Seattle Mariners", "MLB_CHC": "Chicago Cubs", "MLB_BOS": "Boston Red Sox",
        "MLB_CLE": "Cleveland Guardians", "MLB_OAK": "Oakland Athletics", "MLB_WSH": "Washington Nationals",
        "MLB_MIN": "Minnesota Twins", "MLB_ARI": "Arizona Diamondbacks", "MLB_COL": "Colorado Rockies",
        "MLB_KC": "Kansas City Royals", "MLB_CIN": "Cincinnati Reds", "MLB_PIT": "Pittsburgh Pirates",
        "MLB_TBR": "Tampa Bay Rays", "MLB_SFG": "San Francisco Giants", "MLB_NYM": "New York Mets",
        "DET": "Detroit Tigers", "MIA": "Miami Marlins", "CWS": "Chicago White Sox",
        "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "SD": "San Diego Padres",
        "NYY": "New York Yankees", "TOR": "Toronto Blue Jays", "STL": "St. Louis Cardinals",
        "MIL": "Milwaukee Brewers", "ATL": "Atlanta Braves", "TEX": "Texas Rangers",
        "HOU": "Houston Astros", "BAL": "Baltimore Orioles", "PHI": "Philadelphia Phillies",
        "SEA": "Seattle Mariners", "CHC": "Chicago Cubs", "BOS": "Boston Red Sox",
        "CLE": "Cleveland Guardians", "OAK": "Oakland Athletics", "WSH": "Washington Nationals",
        "MIN": "Minnesota Twins", "ARI": "Arizona Diamondbacks", "COL": "Colorado Rockies",
        "KC": "Kansas City Royals", "CIN": "Cincinnati Reds", "PIT": "Pittsburgh Pirates",
        "TBR": "Tampa Bay Rays", "SFG": "San Francisco Giants", "NYM": "New York Mets"
    }
    
    df, participation_rate = summarize_sport("KXMLBGAME", "MLB", TEAM_MAP)
    return df

# --- Main Execution ---
eastern = pytz.timezone("US/Eastern")
today = datetime.now(eastern).date()
yesterday = today - timedelta(days=1)
start_date = date(2025, 6, 14)
days_since_start = (today - start_date).days
cache = read_bankroll_cache()
start_balance = 850
yesterday_balance = cache.get(str(yesterday))
today_balance = cache.get(str(today))

if today_balance is None or yesterday_balance is None or days_since_start <= 0:
    print("❌ Not enough data to calculate daily change or CAGR.")
    exit()

# Daily Change
pnl = today_balance - yesterday_balance
pct_change = (pnl / yesterday_balance) * 100

# CAGR (normalized for capital injection)
actual_total_capital = start_balance + 1100  # $850 + $1100 = $1950 (July 8th injection)
cagr = (today_balance / actual_total_capital) ** (1 / days_since_start) - 1

# Simulate distribution with 1.3% fee per trade and July 8th capital injection
decimal_odds = 100 / 186 + 1
break_even_win_prob = 1 / decimal_odds
b = decimal_odds - 1

capital_injection_day = 24
capital_injection_amount = 1100
july_23_cutoff_day = (date(2025, 7, 23) - start_date).days  # Day when edge changes from 1% to 5%

np.random.seed(42)
final_bankrolls = []
total_capital_injected = []

for _ in range(10000):
    bank = start_balance
    total_injected = 0
    
    for day in range(days_since_start):
        if day == capital_injection_day:
            bank += capital_injection_amount
            total_injected += capital_injection_amount
        
        if day < july_23_cutoff_day:
            edge_multiplier = 1.01  # 1% edge
        else:
            edge_multiplier = 1.04  # 4% edge
            
        true_win_prob = break_even_win_prob * edge_multiplier
        p = true_win_prob
        
        kelly_fraction = (b * p - (1 - p)) / b
        adjusted_kelly = 0.75 * kelly_fraction
            
        day_bank = bank
        for _ in range(7):
            wager = adjusted_kelly * day_bank
            fee = 0.013 * wager
            if np.random.rand() < p:
                bank += wager * b - fee
            else:
                bank -= wager + fee
                
    final_bankrolls.append(bank)
    total_capital_injected.append(total_injected)

current_day = days_since_start - 1
if current_day < july_23_cutoff_day:
    current_edge_multiplier = 1.01
else:
    current_edge_multiplier = 1.05

current_true_win_prob = break_even_win_prob * current_edge_multiplier
expected_value_per_dollar = (current_true_win_prob * b) - ((1 - current_true_win_prob) * 1) - 0.013
print(f"Expected Value per $1 bet (current edge {(current_edge_multiplier-1)*100:.0f}%): ${expected_value_per_dollar:.4f}")

final_bankrolls = np.array(final_bankrolls)
total_capital_injected = np.array(total_capital_injected)
total_capital_base = start_balance + total_capital_injected

sim_returns_raw = (final_bankrolls / total_capital_base)
sim_cagrs = np.full_like(sim_returns_raw, np.nan)
valid_mask = sim_returns_raw > 0
sim_cagrs[valid_mask] = sim_returns_raw[valid_mask] ** (1 / days_since_start) - 1

actual_cagr_normalized = cagr
percentile = np.sum(sim_cagrs <= actual_cagr_normalized) / np.sum(~np.isnan(sim_cagrs)) * 100

# Total return since start (normalized for capital injection)
total_return_pct_raw = (today_balance / start_balance - 1) * 100
total_return_pct_normalized = (today_balance / actual_total_capital - 1) * 100

sport_summaries = {}
sport_configs = {
    "MLB": ("KXMLBGAME", {
        "MLB_DET": "Detroit Tigers", "MLB_MIA": "Miami Marlins", "MLB_CWS": "Chicago White Sox",
        "MLB_LAA": "Los Angeles Angels", "MLB_LAD": "Los Angeles Dodgers", "MLB_SD": "San Diego Padres",
        "MLB_NYY": "New York Yankees", "MLB_TOR": "Toronto Blue Jays", "MLB_STL": "St. Louis Cardinals",
        "MLB_MIL": "Milwaukee Brewers", "MLB_ATL": "Atlanta Braves", "MLB_TEX": "Texas Rangers",
        "MLB_HOU": "Houston Astros", "MLB_BAL": "Baltimore Orioles", "MLB_PHI": "Philadelphia Phillies",
        "MLB_SEA": "Seattle Mariners", "MLB_CHC": "Chicago Cubs", "MLB_BOS": "Boston Red Sox",
        "MLB_CLE": "Cleveland Guardians", "MLB_OAK": "Oakland Athletics", "MLB_WSH": "Washington Nationals",
        "MLB_MIN": "Minnesota Twins", "MLB_ARI": "Arizona Diamondbacks", "MLB_COL": "Colorado Rockies",
        "MLB_KC": "Kansas City Royals", "MLB_CIN": "Cincinnati Reds", "MLB_PIT": "Pittsburgh Pirates",
        "MLB_TBR": "Tampa Bay Rays", "MLB_SFG": "San Francisco Giants", "MLB_NYM": "New York Mets",
        "DET": "Detroit Tigers", "MIA": "Miami Marlins", "CWS": "Chicago White Sox",
        "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "SD": "San Diego Padres",
        "NYY": "New York Yankees", "TOR": "Toronto Blue Jays", "STL": "St. Louis Cardinals",
        "MIL": "Milwaukee Brewers", "ATL": "Atlanta Braves", "TEX": "Texas Rangers",
        "HOU": "Houston Astros", "BAL": "Baltimore Orioles", "PHI": "Philadelphia Phillies",
        "SEA": "Seattle Mariners", "CHC": "Chicago Cubs", "BOS": "Boston Red Sox",
        "CLE": "Cleveland Guardians", "OAK": "Oakland Athletics", "WSH": "Washington Nationals",
        "MIN": "Minnesota Twins", "ARI": "Arizona Diamondbacks", "COL": "Colorado Rockies",
        "KC": "Kansas City Royals", "CIN": "Cincinnati Reds", "PIT": "Pittsburgh Pirates",
        "TBR": "Tampa Bay Rays", "SFG": "San Francisco Giants", "NYM": "New York Mets"
    })
}

sport_html_sections = []
for sport_name, (series_ticker, team_map) in sport_configs.items():
    try:
        df, participation_rate = summarize_sport(series_ticker, sport_name, team_map)
        html = df.to_html(index=False, border=0, justify="center")
        sport_html_sections.append(f"""
        <div class="section-title">⚾ {sport_name} Strategy</div>
        <div class="metric">Participation Rate: <b>{participation_rate}</b></div>
        {html}
        """)
    except Exception as e:
        print(f"Error generating {sport_name} summary: {e}")

mlb_df = summarize_mlb()
mlb_html = mlb_df.to_html(index=False, border=0, justify="center")
bet_on_games = len(mlb_df[mlb_df["team"] != "TOTAL"])

# HTML Email
open_summary = get_open_positions_from_yesterday()

email_body = f"""
<html>
<head>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f8f9fa; padding: 20px; }}
    .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    .section-title {{ font-size: 18px; font-weight: bold; margin-top: 20px; }}
    .metric {{ font-size: 16px; margin: 5px 0; }}
    .positive {{ color: green; }}
    .negative {{ color: red; }}
    table {{ margin-top: 10px; border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
    th {{ background-color: #f2f2f2; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>📊 Kalshi Daily Report — {today}</h2>

    {''.join(sport_html_sections)}

    <div class="section-title">💰 Account Summary</div>
    <div class="metric">Starting Balance: <b>${yesterday_balance:,.2f}</b></div>
    <div class="metric">Current Balance: <b>${today_balance:,.2f}</b></div>
    <div class="metric {'negative' if pnl < 0 else 'positive'}">Day-over-Day Change: <b>${pnl:,.2f} ({pct_change:.2f}%)</b></div>

    <div class="section-title">📂 Open Positions from Yesterday's Buys</div>
    <div class="metric">{'<br>'.join(open_summary) if open_summary else 'None'}</div>

    <div class="section-title">📈 Performance Since 6/14/25 ({days_since_start} Days)</div>
    <div class="metric">Total Return: <b>{total_return_pct_normalized:.2f}%</b></div>
    <div class="metric">Daily CAGR: <b>{cagr * 100:.2f}%</b></div>
    <div class="metric">Percentile Rank vs Strategy (w/ 1.3% Fee/Trade): <b>{percentile:.1f}%</b></div>
    <div class="metric"><i>Simulation assumes 7 trades/day with 1.3% fee on each trade.</i></div>


  </div>
</body>
</html>
"""

# Send email
msg = MIMEText(email_body, "html")
msg["Subject"] = f"📊 Kalshi Daily Report — {today}"
msg["From"] = sender_email
msg["To"] = ", ".join(receiver_email)

try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, app_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()
    print("✅ Email sent successfully!")
except Exception as e:
    print(f"❌ Failed to send email: {e}")
