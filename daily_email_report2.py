import json
import os
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
import smtplib
import pytz

# 📅 Date setup
eastern = pytz.timezone('US/Eastern')
today = datetime.now(eastern).date()
yesterday = today - pd.Timedelta(days=1)

# 📂 Cache location
filename = "/home/walkwalkm1/bankroll_cache.json"
today_str = str(today)
yesterday_str = str(yesterday)

# 🔢 Load balances
today_balance = 0
yesterday_balance = 0

if os.path.exists(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        today_balance = data.get(today_str, 0)
        yesterday_balance = data.get(yesterday_str, 0)

# 📈 Percent change
pct_change = ((today_balance - yesterday_balance) / yesterday_balance * 100) if yesterday_balance else 0

# ✉️ Email message
body = f"""

📅 Date: {today}
💵 Today’s Balance: ${today_balance:.2f}
📉 Yesterday’s Balance: ${yesterday_balance:.2f}
📊 Day-over-Day Change: {pct_change:.2f}%
"""

# 🔧 Uses local sendmail (you can swap for Gmail SMTP if needed)
# ✉️ Use Gmail SMTP
recipients = ["walkwalkm1@gmail.com","robindu1999@gmail.com"]

gmail_user = "walkwalkm1@gmail.com"
gmail_app_password = "upcfsvrhavhyxtiy"  # 🔒 Replace this!

msg = MIMEText(body)
msg["From"] = gmail_user
msg["To"] = ", ".join(recipients)

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_app_password)
        server.sendmail(msg["From"], recipients, msg.as_string())
    print("✅ Email sent successfully.")
except Exception as e:
    print(f"❌ Email failed: {e}")
