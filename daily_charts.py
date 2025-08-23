#!/usr/bin/env python3
"""
Daily Charts Script for Kalshi Trading Strategy

Generates time-series charts for the previous day's games showing:
- Kalshi implied odds vs composite devigged sportsbook odds over time
- Bet placement times and actual outcomes
- Convergence analysis and market efficiency metrics

Sends separate email with daily charts. Run daily after games complete.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
from datetime import datetime, timedelta, date
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Dict, List, Optional

from diagnostics_utils import (
    load_odds_timeseries_data, load_placed_orders, load_game_results,
    save_chart_for_email
)

try:
    from credentials import SENDER_EMAIL, EMAIL_APP_PASSWORD
    RECEIVER_EMAILS = ["walkwalkm1@gmail.com"]
    EMAIL_ENABLED = True
except ImportError:
    print("Email credentials not found - will save charts locally only")
    EMAIL_ENABLED = False

eastern = pytz.timezone("US/Eastern")

class DailyChartsGenerator:
    def __init__(self, target_date: Optional[str] = None):
        if target_date is None:
            yesterday = datetime.now(eastern).date() - timedelta(days=1)
            self.target_date = yesterday.strftime('%Y-%m-%d')
        else:
            self.target_date = target_date
        
        print(f"📅 Generating charts for {self.target_date}")
        
        self.odds_data = load_odds_timeseries_data(
            start_date=self.target_date, 
            end_date=self.target_date
        )
        self.orders_data = load_placed_orders()
        self.results_data = load_game_results()
        
        if not self.orders_data.empty:
            self.orders_data['date'] = pd.to_datetime(self.orders_data['timestamp']).dt.date
            self.orders_data = self.orders_data[
                self.orders_data['date'] == pd.to_datetime(self.target_date).date()
            ]
        
        self.charts = []
        
    def create_odds_convergence_chart(self, game_data: pd.DataFrame, game_title: str) -> Optional[str]:
        """Create time-series chart for a single game showing odds convergence"""
        if game_data.empty:
            return None
        
        game_data = game_data.sort_values('timestamp')
        game_data['timestamp'] = pd.to_datetime(game_data['timestamp'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(game_data['timestamp'], game_data['kalshi_implied_odds'], 
                'b-', linewidth=2, label='Kalshi Implied Odds', marker='o', markersize=4)
        
        ax.plot(game_data['timestamp'], game_data['composite_devigged_odds'], 
                'r-', linewidth=2, label='Composite Devigged Sportsbook Odds', marker='s', markersize=4)
        
        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        color_idx = 0
        
        for _, row in game_data.iterrows():
            per_book_odds = row.get('per_book_american_odds', {})
            per_book_probs = row.get('per_book_implied_prob', {})
            
            if per_book_probs:
                for book, prob in per_book_probs.items():
                    if book not in [line.get_label() for line in ax.lines]:
                        ax.scatter(row['timestamp'], prob, 
                                 c=colors[color_idx % len(colors)], 
                                 alpha=0.6, s=30, label=f'{book}')
                        color_idx += 1
        
        if not self.orders_data.empty:
            game_orders = self.orders_data[self.orders_data['team'].isin(game_data['team'].unique())]
            for _, order in game_orders.iterrows():
                ax.axvline(pd.to_datetime(order['timestamp']), 
                          color='black', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(pd.to_datetime(order['timestamp']), 0.9, 
                       f"BET: {order['team']}\n{order['price']}¢", 
                       rotation=90, verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        game_results = self.results_data[
            (self.results_data['date'] == self.target_date) &
            (self.results_data['home_team'].isin(game_data['team'].unique()) |
             self.results_data['away_team'].isin(game_data['team'].unique()))
        ]
        
        if not game_results.empty:
            result = game_results.iloc[0]
            winner = result.get('winner_team', 'Unknown')
            score_text = f"Final: {result.get('home_team', 'Home')} {result.get('home_score', 0)} - {result.get('away_score', 0)} {result.get('away_team', 'Away')}"
            ax.text(0.02, 0.98, f"WINNER: {winner}\n{score_text}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Implied Probability')
        ax.set_title(f'{game_title} - Odds Convergence Analysis\n{self.target_date}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        safe_title = "".join(c for c in game_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        chart_path = save_chart_for_email(fig, f"convergence_{safe_title}_{self.target_date}.png")
        
        return chart_path
    
    def analyze_convergence_metrics(self, game_data: pd.DataFrame) -> Dict:
        """Calculate convergence metrics for a game"""
        if len(game_data) < 2:
            return {}
        
        game_data = game_data.sort_values('timestamp')
        
        correlation = np.corrcoef(
            game_data['kalshi_implied_odds'], 
            game_data['composite_devigged_odds']
        )[0, 1]
        
        kalshi_range = game_data['kalshi_implied_odds'].max() - game_data['kalshi_implied_odds'].min()
        sportsbook_range = game_data['composite_devigged_odds'].max() - game_data['composite_devigged_odds'].min()
        
        final_spread = abs(
            game_data['kalshi_implied_odds'].iloc[-1] - 
            game_data['composite_devigged_odds'].iloc[-1]
        )
        
        kalshi_changes = game_data['kalshi_implied_odds'].diff().abs().sum()
        sportsbook_changes = game_data['composite_devigged_odds'].diff().abs().sum()
        
        return {
            'correlation': correlation,
            'kalshi_volatility': kalshi_range,
            'sportsbook_volatility': sportsbook_range,
            'final_spread': final_spread,
            'kalshi_total_movement': kalshi_changes,
            'sportsbook_total_movement': sportsbook_changes,
            'more_volatile': 'Kalshi' if kalshi_changes > sportsbook_changes else 'Sportsbooks'
        }
    
    def create_summary_chart(self, all_metrics: List[Dict]) -> Optional[str]:
        """Create summary chart of convergence metrics across all games"""
        if not all_metrics:
            return None
        
        metrics_df = pd.DataFrame(all_metrics)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.hist(metrics_df['correlation'].dropna(), bins=10, alpha=0.7, edgecolor='black')
        ax1.set_title('Kalshi-Sportsbook Correlation Distribution')
        ax1.set_xlabel('Correlation')
        ax1.set_ylabel('Number of Games')
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(metrics_df['kalshi_volatility'], metrics_df['sportsbook_volatility'], alpha=0.7)
        ax2.plot([0, metrics_df[['kalshi_volatility', 'sportsbook_volatility']].max().max()], 
                [0, metrics_df[['kalshi_volatility', 'sportsbook_volatility']].max().max()], 
                'r--', alpha=0.7, label='Equal Volatility')
        ax2.set_xlabel('Kalshi Volatility')
        ax2.set_ylabel('Sportsbook Volatility')
        ax2.set_title('Volatility Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.hist(metrics_df['final_spread'].dropna(), bins=10, alpha=0.7, edgecolor='black')
        ax3.set_title('Final Odds Spread Distribution')
        ax3.set_xlabel('Final Spread (Kalshi - Sportsbook)')
        ax3.set_ylabel('Number of Games')
        ax3.grid(True, alpha=0.3)
        
        leadership_counts = metrics_df['more_volatile'].value_counts()
        ax4.pie(leadership_counts.values, labels=leadership_counts.index, autopct='%1.1f%%')
        ax4.set_title('Market Movement Leadership')
        
        plt.suptitle(f'Daily Convergence Analysis Summary - {self.target_date}', fontsize=16)
        plt.tight_layout()
        
        chart_path = save_chart_for_email(fig, f"daily_summary_{self.target_date}.png")
        return chart_path
    
    def generate_daily_charts(self):
        """Generate all charts for the target date"""
        if self.odds_data.empty:
            print(f"⚠️ No odds data found for {self.target_date}")
            return
        
        print(f"📊 Found {len(self.odds_data)} odds records for {self.target_date}")
        
        games = {}
        all_metrics = []
        
        for sport in self.odds_data['sport'].unique():
            sport_data = self.odds_data[self.odds_data['sport'] == sport]
            
            for team in sport_data['team'].unique():
                team_data = sport_data[sport_data['team'] == team]
                
                if len(team_data) >= 2:  # Need at least 2 data points for a chart
                    game_title = f"{sport}: {team}"
                    games[game_title] = team_data
        
        print(f"📈 Creating charts for {len(games)} games...")
        
        for game_title, game_data in games.items():
            print(f"  📊 Creating chart for {game_title}")
            
            chart_path = self.create_odds_convergence_chart(game_data, game_title)
            if chart_path:
                self.charts.append(chart_path)
            
            metrics = self.analyze_convergence_metrics(game_data)
            if metrics:
                metrics['game'] = game_title
                all_metrics.append(metrics)
        
        if all_metrics:
            print("📊 Creating summary chart...")
            summary_chart = self.create_summary_chart(all_metrics)
            if summary_chart:
                self.charts.append(summary_chart)
        
        print(f"✅ Generated {len(self.charts)} charts")
        
        return all_metrics
    
    def generate_html_report(self, metrics: List[Dict]) -> str:
        """Generate HTML report for daily charts"""
        
        if metrics:
            avg_correlation = np.mean([m['correlation'] for m in metrics if not np.isnan(m['correlation'])])
            avg_final_spread = np.mean([m['final_spread'] for m in metrics])
            kalshi_leadership = sum(1 for m in metrics if m['more_volatile'] == 'Kalshi')
            total_games = len(metrics)
        else:
            avg_correlation = 0
            avg_final_spread = 0
            kalshi_leadership = 0
            total_games = 0
        
        html_report = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h3 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 Daily Charts Report - {self.target_date}</h1>
                <p><em>Time-series analysis of odds convergence and market efficiency</em></p>
                
                <h3>📊 Summary Metrics</h3>
                <div class="metric">
                    <strong>Games Analyzed:</strong> <span class="metric-value">{total_games}</span>
                </div>
                <div class="metric">
                    <strong>Average Correlation:</strong> <span class="metric-value">{avg_correlation:.3f}</span>
                    <br><small>Higher correlation indicates more efficient price discovery</small>
                </div>
                <div class="metric">
                    <strong>Average Final Spread:</strong> <span class="metric-value">{avg_final_spread:.3f}</span>
                    <br><small>Lower spread indicates better market convergence</small>
                </div>
                <div class="metric">
                    <strong>Market Leadership:</strong> <span class="metric-value">Kalshi: {kalshi_leadership}/{total_games}</span>
                    <br><small>Number of games where Kalshi showed more price movement</small>
                </div>
                
                <h3>📈 Individual Game Analysis</h3>
                <p>Each chart shows the time-series evolution of odds throughout the day, with:</p>
                <ul>
                    <li><strong>Blue line:</strong> Kalshi implied odds</li>
                    <li><strong>Red line:</strong> Composite devigged sportsbook odds</li>
                    <li><strong>Colored dots:</strong> Individual sportsbook odds</li>
                    <li><strong>Black dashed lines:</strong> Bet placement times</li>
                    <li><strong>Green box:</strong> Final game result</li>
                </ul>
                
                <h3>🔍 Convergence Analysis</h3>
                <p>The charts reveal market efficiency patterns:</p>
                <ul>
                    <li><strong>High correlation:</strong> Markets are efficiently pricing events</li>
                    <li><strong>Low final spread:</strong> Good price discovery by game time</li>
                    <li><strong>Volatility patterns:</strong> Which market leads price discovery</li>
                </ul>
                
                <p><em>Charts generated on {datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')}</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_report
    
    def save_and_send_charts(self, metrics: List[Dict]):
        """Save charts locally and send via email if configured"""
        html_report = self.generate_html_report(metrics)
        
        report_filename = f"daily_charts_{self.target_date}.html"
        with open(report_filename, 'w') as f:
            f.write(html_report)
        print(f"✅ Report saved to {report_filename}")
        
        if EMAIL_ENABLED and self.charts:
            try:
                msg = MIMEMultipart()
                msg['Subject'] = f"📈 Daily Charts Report - {self.target_date}"
                msg['From'] = SENDER_EMAIL
                msg['To'] = ", ".join(RECEIVER_EMAILS)
                
                msg.attach(MIMEText(html_report, 'html'))
                
                for chart_path in self.charts:
                    if os.path.exists(chart_path):
                        with open(chart_path, 'rb') as f:
                            img_data = f.read()
                        img = MIMEImage(img_data)
                        img.add_header('Content-Disposition', f'attachment; filename={os.path.basename(chart_path)}')
                        msg.attach(img)
                
                # Send email
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(SENDER_EMAIL, EMAIL_APP_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
                server.quit()
                print("✅ Daily charts email sent successfully!")
                
            except Exception as e:
                print(f"❌ Failed to send email: {e}")
        else:
            if not EMAIL_ENABLED:
                print("📧 Email not configured - charts saved locally only")
            if not self.charts:
                print("📊 No charts generated to send")
    
    def run_daily_analysis(self):
        """Run complete daily charts analysis"""
        print(f"📈 Starting daily charts analysis for {self.target_date}...")
        
        metrics = self.generate_daily_charts()
        
        if self.charts:
            print("📧 Saving and sending charts...")
            self.save_and_send_charts(metrics)
        else:
            print("⚠️ No charts generated - insufficient data")
        
        print("✅ Daily charts analysis complete!")

def main():
    import sys
    
    target_date = None
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    
    generator = DailyChartsGenerator(target_date)
    generator.run_daily_analysis()

if __name__ == "__main__":
    main()
