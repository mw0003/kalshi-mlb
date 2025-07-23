import requests
import pandas as pd
import pytz
import statsapi
from datetime import datetime
import base64
import json
import time as time_module
import uuid
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from IPython.display import display
import os


# 🔐 Kalshi Credentials
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

# 🔑 Load RSA Key
private_key = serialization.load_pem_private_key(
    RSA_PRIVATE_KEY_PEM.encode(), password=None, backend=default_backend()
)

def sign_request(method, path, key_id, private_key):
    timestamp = str(int(time_module.time() * 1000))
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

# 💰 Get Kalshi Balance
def get_balance():
    path = "/trade-api/v2/portfolio/balance"
    url = f"https://api.elections.kalshi.com{path}"
    headers = sign_request("GET", path, API_KEY, private_key)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("balance", 0) / 100

# 🕒 Date & bankroll
eastern = pytz.timezone('US/Eastern')
today = datetime.now(eastern).date()

def get_or_cache_daily_bankroll():
    filename = "/home/walkwalkm1/bankroll_cache.json"
    today_str = str(today)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        if today_str in data:
            return data[today_str]
    bankroll = get_balance()
    data = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    data[today_str] = bankroll
    with open(filename, "w") as f:
        json.dump(data, f)
    return bankroll

bankroll = get_or_cache_daily_bankroll()
print(f"\n💵 Kalshi Balance: ${bankroll:.2f}")

# 📈 Kalshi & FanDuel data
def fetch_kalshi_mlb_odds_active_only():
    url = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMLBGAME"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    rows = []
    for market in data.get("markets", []):
        if market.get("status") != "active":
            continue
        parts = market.get("ticker", "").split('-')
        if len(parts) < 3:
            continue
        rows.append({
            "Market Ticker": market.get("ticker"),
            "Game Title": market.get("title"),
            "Team": parts[-1],
            "Kalshi YES Ask (¢)": market.get("yes_ask")
        })
    return pd.DataFrame(rows)

def fetch_fanduel_odds(api_key):
    url = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds'
    params = {
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": "fanduel",
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    odds_lookup = {}
    for game in response.json():
        start_time = pd.to_datetime(game.get("commence_time"), utc=True).tz_convert('US/Eastern')
        if start_time.date() != today:
            continue
        for b in game.get("bookmakers", []):
            if b["key"] == "fanduel":
                market = next((m for m in b.get("markets", []) if m["key"] == "h2h"), None)
                if market:
                    for outcome in market.get("outcomes", []):
                        team = outcome["name"].strip().replace("Oakland Athletics", "Athletics")
                        odds_lookup[team] = outcome["price"]
    return odds_lookup

def build_opponent_map():
    games = statsapi.schedule(start_date=str(today), end_date=str(today))
    matchup = {}
    for game in games:
        away = game['away_name'].replace("Oakland Athletics", "Athletics")
        home = game['home_name'].replace("Oakland Athletics", "Athletics")
        matchup[away] = home
        matchup[home] = away
    return matchup

# 🔢 Kelly

def kelly_wager(fair_odds, your_odds, bankroll):
    try:
        if pd.isna(fair_odds) or pd.isna(your_odds):
            return 0
        return max((((1 / fair_odds) * (your_odds - 1)) - (1 - (1 / fair_odds))) / (your_odds - 1) * bankroll, 0)
    except:
        return 0

# Team abbreviation map
team_abbr_to_name = {
    "ATL": "Atlanta Braves", "AZ": "Arizona Diamondbacks", "DET": "Detroit Tigers", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CLE": "Cleveland Guardians", "KC": "Kansas City Royals", "HOU": "Houston Astros",
    "NYM": "New York Mets", "WSH": "Washington Nationals", "PHI": "Philadelphia Phillies", "CHC": "Chicago Cubs",
    "SF": "San Francisco Giants", "TEX": "Texas Rangers", "SEA": "Seattle Mariners", "MIA": "Miami Marlins",
    "CWS": "Chicago White Sox", "MIN": "Minnesota Twins", "LAA": "Los Angeles Angels", "NYY": "New York Yankees",
    "TOR": "Toronto Blue Jays", "PIT": "Pittsburgh Pirates", "LAD": "Los Angeles Dodgers",
    "MIL": "Milwaukee Brewers", "STL": "St. Louis Cardinals", "COL": "Colorado Rockies",
    "CIN": "Cincinnati Reds", "SD": "San Diego Padres", "TB": "Tampa Bay Rays", "OAK": "Oakland Athletics", "ATH": "Athletics"
}

kalshi_df = fetch_kalshi_mlb_odds_active_only()
fanduel_odds = fetch_fanduel_odds("141e7d4fb0c345a19225eb2f2b114273")
opponent_map = build_opponent_map()

kalshi_df["Team Name"] = kalshi_df["Team"].map(team_abbr_to_name)
kalshi_df["Opponent Name"] = kalshi_df["Team Name"].map(opponent_map)
kalshi_df["FanDuel Odds (American)"] = kalshi_df["Team Name"].map(fanduel_odds)

kalshi_df["Kalshi %"] = kalshi_df["Kalshi YES Ask (¢)"] / 100
kalshi_df["Decimal Odds (Kalshi)"] = 1 / kalshi_df["Kalshi %"]
kalshi_df["Decimal Odds (FanDuel)"] = kalshi_df["FanDuel Odds (American)"].apply(
    lambda x: (x / 100) + 1 if x > 0 else (100 / -x) + 1 if pd.notna(x) else None
)

raw_edge = kalshi_df["Decimal Odds (Kalshi)"] * (1 / kalshi_df["Decimal Odds (FanDuel)"]) - 1
kalshi_df["% Edge"] = raw_edge.apply(lambda x: f"{round(x * 100, 1)}%" if pd.notna(x) else None)
kalshi_df["numeric_edge"] = raw_edge

kalshi_df["$ Wager"] = kalshi_df.apply(
    lambda row: (
        f"${round(min(kelly_wager(row['Decimal Odds (FanDuel)'], row['Decimal Odds (Kalshi)'], bankroll), bankroll * 0.3))}"
        if pd.notna(row["Decimal Odds (FanDuel)"]) and pd.notna(row["Decimal Odds (Kalshi)"])
        else "$0"
    ),
    axis=1
)

kalshi_df["Kalshi YES Ask (¢)"] = kalshi_df["Kalshi YES Ask (¢)"].astype(int)
kalshi_df = kalshi_df.sort_values(by="numeric_edge", ascending=False).drop(columns=["numeric_edge"]).reset_index(drop=True)

final_df = kalshi_df[[
    "Team Name", "Opponent Name",
    "Kalshi YES Ask (¢)", "FanDuel Odds (American)",
    "% Edge", "$ Wager", "Market Ticker"
]]

#print("\n📊 Full Table:")
#display(final_df)

filtered_df = final_df[
    (final_df["Kalshi YES Ask (¢)"] >= 60) &
    (final_df["Kalshi YES Ask (¢)"] <= 95) &
    (final_df["% Edge"].str.replace('%', '').astype(float) >= 5) &
    (final_df["% Edge"].str.replace('%', '').astype(float) < 9.1)
].reset_index(drop=True)

seen_teams = set()
filtered_cleaned = []
for _, row in filtered_df.iterrows():
    team = row["Team Name"]
    opponent = row["Opponent Name"]
    if opponent in seen_teams:
        continue
    seen_teams.add(team)
    filtered_cleaned.append(row)

filtered_df = pd.DataFrame(filtered_cleaned).reset_index(drop=True)

# 📥 Get today's orders (Eastern Time aware)
def get_todays_orders():
    path = "/trade-api/v2/portfolio/orders"
    url = f"https://api.elections.kalshi.com{path}"
    headers = sign_request("GET", path, API_KEY, private_key)

    from datetime import time  # needed for time.min / time.max
    today_start_dt = eastern.localize(datetime.combine(today, time.min))
    today_end_dt = eastern.localize(datetime.combine(today, time.max))

    today_start = int(today_start_dt.timestamp())
    today_end = int(today_end_dt.timestamp())

    params = {"min_ts": today_start, "max_ts": today_end}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("orders", [])

# 🚫 Exclude previously bet games (buy-only)
executed_team_abbrs = set()
for order in get_todays_orders():
    if (
        order.get("status") in ("executed","resting") and
        order.get("action") == "buy" and
        order.get("ticker")
    ):
        parts = order["ticker"].split('-')
        if len(parts) >= 3:
            abbr = parts[-1]
            executed_team_abbrs.add(abbr)

# 🔍 Debug print to verify which teams are being excluded
print(f"✅ Executed team abbreviations (BUY only): {sorted(executed_team_abbrs)}")

# Map to full team names
executed_team_names = {team_abbr_to_name[abbr] for abbr in executed_team_abbrs if abbr in team_abbr_to_name}
print(f"✅ Full team names we already bet on: {sorted(executed_team_names)}")

# Identify rows being filtered out
if not filtered_df.empty:
    flagged_rows = filtered_df[
        filtered_df["Team Name"].isin(executed_team_names) |
        filtered_df["Opponent Name"].isin(executed_team_names)
    ]

    print(f"\n🚫 Rows removed due to duplicate or opponent bets: {len(flagged_rows)}")
    if not flagged_rows.empty:
        display(flagged_rows[["Team Name", "Opponent Name", "Market Ticker"]])

    # Apply the actual filtering
    filtered_df = filtered_df[
        ~filtered_df["Team Name"].isin(executed_team_names) &
        ~filtered_df["Opponent Name"].isin(executed_team_names)
    ].reset_index(drop=True)

    print("\n📈 Final filtered_df after exclusions:")
    display(filtered_df)
else:
    print("⚠️ filtered_df is empty before exclusions.")

# 🛒 Submit orders
def submit_order(market_ticker, side, quantity, price):
    path = "/trade-api/v2/portfolio/orders"
    url = f"https://api.elections.kalshi.com{path}"
    headers = sign_request("POST", path, API_KEY, private_key)
    payload = {
        "action": "buy",
        "type": "limit",
        "side": side,
        "ticker": market_ticker,
        "count": quantity,
        "yes_price": price,
        "client_order_id": str(uuid.uuid4())
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

#print("\n🛒 Executing Orders:")
for _, row in filtered_df.iterrows():
    try:
        # Final check using abbreviations
        team_abbr = [abbr for abbr, name in team_abbr_to_name.items() if name == row["Team Name"]]
        opp_abbr = [abbr for abbr, name in team_abbr_to_name.items() if name == row["Opponent Name"]]

        if (team_abbr and team_abbr[0] in executed_team_abbrs) or (opp_abbr and opp_abbr[0] in executed_team_abbrs):
            print(f"⚠️ Skipping {row['Team Name']} vs {row['Opponent Name']} — already bet.")
            continue

        ticker = row["Market Ticker"]
        team = row["Team Name"]
        wager_dollars = 0.75 * float(row["$ Wager"].strip("$") or 0)
        price = int(row["Kalshi YES Ask (¢)"])
        cost_per_contract = price / 100
        suggested_contracts = int(wager_dollars // cost_per_contract)
        contracts = min(suggested_contracts, int(0.2 * bankroll / cost_per_contract))
        total_cost = contracts * cost_per_contract

        if contracts < 1 or total_cost > bankroll:
            #print(f"🚫 {team} — Not enough bankroll to place even 1 contract.")
            continue

        result = submit_order(ticker, "yes", contracts, price)
        print(f"▶️ {team} → {contracts} contracts at {price}¢ → ✅ {result}")
        bankroll -= total_cost

    except Exception as e:
        #print(f"❌ {row['Team Name']} — Error: {e}")
        pass

