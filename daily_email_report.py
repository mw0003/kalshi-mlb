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

# --- Email Configuration ---
sender_email = "walkwalkm1@gmail.com"
receiver_email = ["walkwalkm1@gmail.com","robindu1999@gmail.com","mattkass329@gmail.com"]
app_password = "upcfsvrhavhyxtiy"

# --- Kalshi Credentials ---
API_KEY = "52080fdd-2c8d-43c5-b98b-914399ce5689"
RSA_PRIVATE_KEY_PEM = """-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEApIAcrVGTn1c+Ar2nXZZxEtQYIGG9a6N1XfB6L1MtBARzKuAm
aOH7lyj0j4MRIyTjw5a7vcaRGDbhB3WoFQcoGSpjFdkRq36wXtylwhsBSm6BEvWE
PzGhriMo5Ch2P4BQtkVKFvWJfANCxW4a8KWtGBXC/E4486xlb8uyXOffBfvC1JMe
Q2vIxQbf4ftiCYeaTcRl8MsqX1oqvb55Qhl40SXNsEicScVTNje/m49nCDY/C8QP
HnK+ZSnPTY7SfltjSxicvqsZGv3UfX9/UXpGX38OTd3n9+6JAPBnZaXCvoj4f/vl
DfYOFrOEAIl0R51U9UXVubFhvkoybLOQ+PlakwIDAQABAoIBAFc3YXz3Ini57bPQ
T/tLtznPX9dTWvXF3YVn6bBLvjNCFLmnzFWRcy4K1dd9G0nx1hyuP2336JfZCOhG
lk5H1Be7pHtB8p9ldSdmfy/x13ZaLm8Z4vsKWnmURKrrVP6IDsME66pOlo08wVsh
7ICopqR9bTsOUh3HyqRCcJfXjCSEJHLoxCAijb935pGKWN84RmUQHEMf2VjQvnZ/
XogKH5+SD+bKW/iU6DcRpaAuFk9arJCaMbii6mBTagC+ENyv0f+XowsqivlMKWy9
7S7OQ9XkTZLLtPolGVxGnYTZn+Q9IkjRk4vr2gD1QMKbyhFw3CrPu29uSAFt3NTV
KuZ/dyECgYEAx+Qi8Wisp1oLN0dIoaKNhYBXWHqmvJcDHji0U+Ec532lzvfOpy1u
Rrq1SXpdDqmJ9ynqkDqk/lVAKGs4+wUFbX+Z7ugkgoRvAn71PUYtUw3xNsgn6LI7
5lsBq8t9z2qVcYx3CwoVbDn/dd1VrmdttJmQIlU9zdau52zw4jl2jbECgYEA0qza
qHXhQ+XYxeT2tO1DlSwJR2fN4t66tSVuqUMg5GPdmJ1rvWe3399GRzqrGlx/6f7f
Qxa5NZ3scvqHomwP0s2rw4PGy5srQeLIsOuWXuQSRSDmFjdDo7xd7N90QCwjC0Ak
dZln+2ERFHPX4yj2kVLRYPhyBR3TSMxk2+JVqYMCgYBQ+D2bUk5Vv+i5LJvkNYdk
I5e+FHjD/dvaexe4voBJ2SC4FLNWDtYTun/C0tktHknvn8APSmIZUAkcFkrPi7om
H8EIAGsBn4mkFi9a8blcYlJqYWuhG8mdxxGHOHeu9Dqy8zYpd50z6M5tPQn/CpBq
zqWO8r6FScgxoHR2/tXiEQKBgEl7hy0ZKMBxDEJCUZbb5yXB3V6to0+NlpwWeVnK
k092UdWomurOoYERtMalfQbN2sP4ZVFWPLWp5s5X+jU58e76U/33GcDs15K8knm7
QpDIhmLcTcTT8+DJlA1KB5dWjcaf0de+8VjqC3YRzexq3k3kECn9nm+QbqDGwis7
79sXAoGAIi2g5zYO9WdrmXfthiZD8cUhHIGPQmH6/kZcZILdQiMatiBGYPtn2DK0
FoEPzFDZD/fqEke86MDXqbjc7IKrQad+DI9D/rKDvnjM+ZH1wfw2RP7rJfd7EH5w
mhlAMQeQS9CmRiGid7cb9rXfAOHJA6az7qFWzVbPyDI9GAsXlZQ=
-----END RSA PRIVATE KEY-----"""

private_key = serialization.load_pem_private_key(
    RSA_PRIVATE_KEY_PEM.encode(), password=None, backend=default_backend()
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
        with open("/home/walkwalkm1/bankroll_cache.json", "r") as f:
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
    order_headers = sign_request("GET", order_path, API_KEY, private_key)
    orders_resp = requests.get(order_url, headers=order_headers, params={"min_ts": start_ts, "max_ts": end_ts})
    orders = orders_resp.json().get("orders", [])

    bought_tickers = {o["ticker"] for o in orders if o["status"] == "executed" and o["action"] == "buy"}

    pos_path = "/trade-api/v2/portfolio/positions"
    pos_url = f"https://api.elections.kalshi.com{pos_path}"
    pos_headers = sign_request("GET", pos_path, API_KEY, private_key)
    positions = requests.get(pos_url, headers=pos_headers).json().get("positions", [])

    return [
        f"{pos['ticker']} — {pos['side']} {pos['count']} @ {pos['average_price']}¢"
        for pos in positions if pos["count"] > 0 and pos["ticker"] in bought_tickers
    ]

def summarize_mlb():
    from tabulate import tabulate
    import time

    TEAM_MAP = {
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

    def local_sign_request(method, path):
        ts = str(int(time.time() * 1000))
        msg = ts + method + path.split('?')[0]
        sig = private_key.sign(msg.encode(), padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ), hashes.SHA256())
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": API_KEY,
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

    orders = [o for o in get_orders() if o.get("ticker", "").startswith("KXMLBGAME")]
    settlements = get_settlements()

    data = []
    total_wager_raw = 0.0
    total_return_raw = 0.0

    for o in orders:
        code = o["ticker"].split("-")[-1]
        team = TEAM_MAP.get(code, code)
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
            with open("/home/walkwalkm1/placed_orders.json", "r") as f:
                placed_orders_data = json.load(f)
            placed_order = next((order for order in placed_orders_data if order["ticker"] == o["ticker"]), {})
            ev_before = f"{placed_order.get('expected_value_before_devig', 0):.1f}%" if placed_order else "N/A"
            ev_after = f"{placed_order.get('expected_value_after_devig', 0):.1f}%" if placed_order else "N/A"
        except FileNotFoundError:
            ev_before = "N/A"
            ev_after = "N/A"

        data.append({
            "team": team,
            "odds": odds,
            "wager": wager,
            "return": return_str,
            "ev_before_devig": ev_before,
            "ev_after_devig": ev_after
        })

    data.append({
        "team": "TOTAL",
        "odds": "",
        "wager": f"${total_wager_raw:.2f}",
        "return": f"${total_return_raw:.2f}",
        "ev_before_devig": "",
        "ev_after_devig": ""
    })

    df = pd.DataFrame(data, columns=["team", "odds", "wager", "return", "ev_before_devig", "ev_after_devig"])
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
true_win_prob = break_even_win_prob * 1.06
b = decimal_odds - 1
p = true_win_prob

expected_value_per_dollar = (p * b) - ((1 - p) * 1) - 0.013
print(f"Expected Value per $1 bet: ${expected_value_per_dollar:.4f}")

kelly_fraction = (b * p - (1 - p)) / b
adjusted_kelly = 0.75 * kelly_fraction

capital_injection_day = 24
capital_injection_amount = 1100

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
            
        day_bank = bank
        for _ in range(12):
            wager = adjusted_kelly * day_bank
            fee = 0.013 * wager
            if np.random.rand() < p:
                bank += wager * b - fee
            else:
                bank -= wager + fee
                
    final_bankrolls.append(bank)
    total_capital_injected.append(total_injected)

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

# MLB table for email
mlb_df = summarize_mlb()
mlb_html = mlb_df.to_html(index=False, border=0, justify="center")

# Count games bet on vs actual games
bet_on_games = len(mlb_df[mlb_df["team"] != "TOTAL"])
formatted_date = yesterday.strftime("%m/%d/%Y")
sched_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={formatted_date}"
sched_data = requests.get(sched_url).json()
total_games = sched_data["totalGames"]
bet_summary_line = f"We bet on <b>{bet_on_games}/{total_games}</b> MLB games yesterday."

# HTML Email
open_summary = get_open_positions_from_yesterday()
weather_start_date = date(2025, 6, 30)
weather_days = (today - weather_start_date).days + 1

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

    <div class="section-title">⚾ MLB Strategy</div>
    <div class="metric">{bet_summary_line}</div>
    {mlb_html}

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
    <div class="metric"><i>Simulation assumes 12 trades/day with 1.3% fee on each trade.</i></div>

    <div class="section-title">☀️ Weather Strategy</div>
    <div class="metric">Days Collecting Data: <b>{weather_days}/30</b></div>

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

def create_team_charts_email():
    try:
        with open("/home/walkwalkm1/odds_timeseries.json", "r") as f:
            timeseries_data = json.load(f)
    except FileNotFoundError:
        print("No time series data available yet")
        return
    
    if not timeseries_data:
        return
    
    df = pd.DataFrame(timeseries_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    teams = df['team'].unique()
    
    chart_attachments = []
    email_body_parts = []
    
    for team in teams:
        team_data = df[df['team'] == team].sort_values('timestamp')
        
        if len(team_data) < 2:
            continue
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(team_data['timestamp'], team_data['kalshi_implied_odds'], 
                label='Kalshi Implied Odds', color='blue')
        ax1.plot(team_data['timestamp'], team_data['fanduel_devigged_odds'], 
                label='FanDuel Devigged Odds', color='red')
        ax1.set_title(f'{team} - Implied Odds Over Time')
        ax1.set_ylabel('Implied Probability')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(team_data['timestamp'], team_data['expected_value'], 
                label='Expected Value', color='green')
        ax2.set_title(f'{team} - Expected Value Over Time')
        ax2.set_ylabel('Expected Value')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True)
        
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        chart_filename = f'/tmp/{team.replace(" ", "_")}_chart.png'
        plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        chart_attachments.append((chart_filename, team))
        email_body_parts.append(f"<h3>{team}</h3><img src='cid:{team.replace(' ', '_')}'><br><br>")
    
    if not chart_attachments:
        print("No charts to send - insufficient data")
        return
    
    msg = MIMEMultipart('related')
    msg['Subject'] = f"📈 MLB Team Analysis Charts — {datetime.now(eastern).date()}"
    msg['From'] = sender_email
    msg['To'] = "walkwalkm1@gmail.com"
    
    html_body = f"""
    <html>
    <body>
        <h2>📈 MLB Team Analysis - Odds & Expected Value Tracking</h2>
        <p>Daily charts showing Kalshi implied odds, FanDuel devigged odds, and expected value over time.</p>
        {''.join(email_body_parts)}
    </body>
    </html>
    """
    
    msg.attach(MIMEText(html_body, 'html'))
    
    for chart_file, team in chart_attachments:
        with open(chart_file, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-ID', f'<{team.replace(" ", "_")}>')
            msg.attach(img)
        os.remove(chart_file)
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, ["walkwalkm1@gmail.com"], msg.as_string())
        server.quit()
        print("✅ Team charts email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send team charts email: {e}")

create_team_charts_email()
