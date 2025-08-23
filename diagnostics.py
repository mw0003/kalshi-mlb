#!/usr/bin/env python3
"""
Weekly Diagnostics Script for Kalshi Trading Strategy

Analyzes historical data to evaluate and improve trading strategy through:
- Accuracy analysis: Kalshi vs Sportsbooks (Brier score, log loss, calibration)
- Arbitrage detection: Between sportsbooks and Kalshi vs sportsbooks
- Profit analysis: ROI by sport/source, Kelly criterion validation
- Statistical significance testing and correlation analysis

Run weekly to analyze all accumulated historical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta, date
import pytz
from scipy import stats
from typing import Dict, List, Tuple, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from diagnostics_utils import (
    load_odds_timeseries_data, load_placed_orders, load_game_results,
    calculate_brier_score, calculate_log_loss, calculate_calibration,
    detect_arbitrage_opportunities, create_calibration_plot, save_chart_for_email
)

try:
    from credentials import SENDER_EMAIL, EMAIL_APP_PASSWORD
    RECEIVER_EMAILS = ["walkwalkm1@gmail.com"]
    EMAIL_ENABLED = True
except ImportError:
    print("Email credentials not found - will save report locally only")
    EMAIL_ENABLED = False

eastern = pytz.timezone("US/Eastern")

class DiagnosticsAnalyzer:
    def __init__(self):
        self.odds_data = load_odds_timeseries_data()
        self.orders_data = load_placed_orders()
        self.results_data = load_game_results()
        self.report_sections = []
        self.charts = []
        
    def merge_data_with_results(self) -> pd.DataFrame:
        """Merge odds data with game results for accuracy analysis"""
        if self.results_data.empty or self.odds_data.empty:
            print("⚠️ Missing odds or results data for accuracy analysis")
            return pd.DataFrame()
        
        odds_df = self.odds_data.copy()
        odds_df['date'] = pd.to_datetime(odds_df['timestamp']).dt.date
        
        results_df = self.results_data.copy()
        results_df['date'] = pd.to_datetime(results_df['date']).dt.date
        
        merged = odds_df.merge(
            results_df, 
            left_on=['date', 'team'], 
            right_on=['date', 'winner_team'], 
            how='inner'
        )
        
        merged['team_won'] = (merged['team'] == merged['winner_team']).astype(int)
        
        return merged
    
    def analyze_accuracy_kalshi_vs_sportsbooks(self):
        """Compare accuracy of Kalshi vs sportsbooks using Brier score, log loss, calibration"""
        merged_data = self.merge_data_with_results()
        if merged_data.empty:
            self.report_sections.append("<h3>❌ Accuracy Analysis</h3><p>Insufficient data for accuracy analysis</p>")
            return
        
        kalshi_probs = merged_data['kalshi_implied_odds'].values
        sportsbook_probs = merged_data['composite_devigged_odds'].values
        outcomes = merged_data['team_won'].values
        
        valid_mask = (kalshi_probs > 0) & (kalshi_probs < 1) & (sportsbook_probs > 0) & (sportsbook_probs < 1)
        kalshi_probs = kalshi_probs[valid_mask]
        sportsbook_probs = sportsbook_probs[valid_mask]
        outcomes = outcomes[valid_mask]
        
        if len(outcomes) < 10:
            self.report_sections.append("<h3>❌ Accuracy Analysis</h3><p>Insufficient valid data points for analysis</p>")
            return
        
        kalshi_brier = calculate_brier_score(kalshi_probs, outcomes)
        sportsbook_brier = calculate_brier_score(sportsbook_probs, outcomes)
        
        kalshi_logloss = calculate_log_loss(kalshi_probs, outcomes)
        sportsbook_logloss = calculate_log_loss(sportsbook_probs, outcomes)
        
        kalshi_losses = (kalshi_probs - outcomes) ** 2
        sportsbook_losses = (sportsbook_probs - outcomes) ** 2
        t_stat, p_value = stats.ttest_rel(kalshi_losses, sportsbook_losses)
        
        kalshi_accuracy = np.mean((kalshi_probs > 0.5) == outcomes)
        sportsbook_accuracy = np.mean((sportsbook_probs > 0.5) == outcomes)
        
        kalshi_cal_fig = create_calibration_plot(kalshi_probs, outcomes, "Kalshi Calibration")
        sportsbook_cal_fig = create_calibration_plot(sportsbook_probs, outcomes, "Sportsbooks Calibration")
        
        kalshi_cal_path = save_chart_for_email(kalshi_cal_fig, "kalshi_calibration.png")
        sportsbook_cal_path = save_chart_for_email(sportsbook_cal_fig, "sportsbook_calibration.png")
        
        self.charts.extend([kalshi_cal_path, sportsbook_cal_path])
        
        sport_analysis = []
        for sport in merged_data['sport'].unique():
            sport_data = merged_data[merged_data['sport'] == sport]
            if len(sport_data) >= 5:
                sport_kalshi_probs = sport_data['kalshi_implied_odds'].values
                sport_sportsbook_probs = sport_data['composite_devigged_odds'].values
                sport_outcomes = sport_data['team_won'].values
                
                sport_kalshi_brier = calculate_brier_score(sport_kalshi_probs, sport_outcomes)
                sport_sportsbook_brier = calculate_brier_score(sport_sportsbook_probs, sport_outcomes)
                
                sport_analysis.append({
                    'sport': sport,
                    'games': len(sport_data),
                    'kalshi_brier': sport_kalshi_brier,
                    'sportsbook_brier': sport_sportsbook_brier,
                    'better_source': 'Kalshi' if sport_kalshi_brier < sport_sportsbook_brier else 'Sportsbooks'
                })
        
        sport_df = pd.DataFrame(sport_analysis)
        
        significance = "significant" if p_value < 0.05 else "not significant"
        better_overall = "Kalshi" if kalshi_brier < sportsbook_brier else "Sportsbooks"
        
        report = f"""
        <h3>📊 Accuracy Analysis: Kalshi vs Sportsbooks</h3>
        <p><strong>Sample Size:</strong> {len(outcomes)} games with valid probability data</p>
        
        <h4>Overall Metrics</h4>
        <table border="1" style="border-collapse: collapse;">
        <tr><th>Source</th><th>Brier Score</th><th>Log Loss</th><th>Accuracy</th></tr>
        <tr><td>Kalshi</td><td>{kalshi_brier:.4f}</td><td>{kalshi_logloss:.4f}</td><td>{kalshi_accuracy:.3f}</td></tr>
        <tr><td>Sportsbooks</td><td>{sportsbook_brier:.4f}</td><td>{sportsbook_logloss:.4f}</td><td>{sportsbook_accuracy:.3f}</td></tr>
        </table>
        
        <p><strong>Winner:</strong> {better_overall} (lower Brier score is better)</p>
        <p><strong>Statistical Significance:</strong> Difference is {significance} (p={p_value:.4f})</p>
        
        <h4>Breakdown by Sport</h4>
        {sport_df.to_html(index=False, border=1) if not sport_df.empty else '<p>Insufficient data for sport breakdown</p>'}
        """
        
        self.report_sections.append(report)
    
    def analyze_arbitrage_opportunities(self):
        """Analyze historical arbitrage opportunities"""
        arb_data = detect_arbitrage_opportunities(self.odds_data)
        
        if arb_data.empty:
            self.report_sections.append("<h3>🔍 Arbitrage Analysis</h3><p>No arbitrage opportunities detected in historical data</p>")
            return
        
        total_arbs = len(arb_data)
        avg_profit_margin = arb_data['profit_margin'].mean()
        max_profit_margin = arb_data['profit_margin'].max()
        
        sport_arbs = arb_data.groupby('sport').agg({
            'profit_margin': ['count', 'mean', 'max']
        }).round(4)
        sport_arbs.columns = ['Count', 'Avg Profit %', 'Max Profit %']
        
        kalshi_arbs = arb_data[arb_data['book1'] == 'Kalshi']
        sportsbook_arbs = arb_data[arb_data['book1'] != 'Kalshi']
        
        arb_data['timestamp'] = pd.to_datetime(arb_data['timestamp'])
        arb_data['hour'] = arb_data['timestamp'].dt.hour
        hourly_arbs = arb_data.groupby('hour').size()
        
        report = f"""
        <h3>🔍 Arbitrage Analysis</h3>
        <p><strong>Total Opportunities:</strong> {total_arbs}</p>
        <p><strong>Average Profit Margin:</strong> {avg_profit_margin:.2%}</p>
        <p><strong>Maximum Profit Margin:</strong> {max_profit_margin:.2%}</p>
        
        <h4>Arbitrage by Sport</h4>
        {sport_arbs.to_html(border=1)}
        
        <h4>Source Analysis</h4>
        <p><strong>Kalshi vs Sportsbooks:</strong> {len(kalshi_arbs)} opportunities</p>
        <p><strong>Between Sportsbooks:</strong> {len(sportsbook_arbs)} opportunities</p>
        
        <h4>Timing Analysis</h4>
        <p>Most arbitrage opportunities occur during hours: {hourly_arbs.nlargest(3).index.tolist()}</p>
        """
        
        self.report_sections.append(report)
    
    def analyze_profit_performance(self):
        """Analyze profit performance and Kelly criterion validation"""
        if self.orders_data.empty:
            self.report_sections.append("<h3>💰 Profit Analysis</h3><p>No order data available for profit analysis</p>")
            return
        
        bankroll_path = "bankroll_cache.json"
        if not os.path.exists(bankroll_path):
            self.report_sections.append("<h3>💰 Profit Analysis</h3><p>No bankroll data available</p>")
            return
        
        with open(bankroll_path, 'r') as f:
            bankroll_data = json.load(f)
        
        bankroll_df = pd.DataFrame(list(bankroll_data.items()), columns=['date', 'balance'])
        bankroll_df['date'] = pd.to_datetime(bankroll_df['date'])
        bankroll_df = bankroll_df.sort_values('date')
        
        bankroll_df['daily_return'] = bankroll_df['balance'].pct_change()
        bankroll_df['cumulative_return'] = (bankroll_df['balance'] / bankroll_df['balance'].iloc[0]) - 1
        
        total_return = bankroll_df['cumulative_return'].iloc[-1]
        daily_returns = bankroll_df['daily_return'].dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        max_drawdown = (bankroll_df['balance'] / bankroll_df['balance'].cummax() - 1).min()
        
        orders_with_results = self.orders_data.merge(
            self.results_data,
            left_on=['team'],
            right_on=['winner_team'],
            how='inner'
        )
        
        if not orders_with_results.empty:
            win_rate = (orders_with_results['team'] == orders_with_results['winner_team']).mean()
            expected_win_rate = orders_with_results['expected_value_after_devig'].mean() / 100
        else:
            win_rate = 0
            expected_win_rate = 0
        
        sport_performance = []
        for sport in self.orders_data.get('sport', pd.Series()).unique():
            if pd.isna(sport):
                continue
            sport_orders = self.orders_data[self.orders_data.get('sport', pd.Series()) == sport]
            sport_performance.append({
                'sport': sport,
                'orders': len(sport_orders),
                'total_cost': sport_orders['total_cost'].sum(),
                'avg_edge': sport_orders['expected_value_after_devig'].mean()
            })
        
        sport_df = pd.DataFrame(sport_performance)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(bankroll_df['date'], bankroll_df['balance'], linewidth=2)
        ax1.set_title('Account Balance Over Time')
        ax1.set_ylabel('Balance ($)')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(daily_returns * 100, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        performance_chart_path = save_chart_for_email(fig, "performance_analysis.png")
        self.charts.append(performance_chart_path)
        
        report = f"""
        <h3>💰 Profit Analysis</h3>
        <p><strong>Total Return:</strong> {total_return:.2%}</p>
        <p><strong>Sharpe Ratio:</strong> {sharpe_ratio:.2f}</p>
        <p><strong>Maximum Drawdown:</strong> {max_drawdown:.2%}</p>
        <p><strong>Win Rate:</strong> {win_rate:.2%} (Expected: {expected_win_rate:.2%})</p>
        
        <h4>Performance by Sport</h4>
        {sport_df.to_html(index=False, border=1) if not sport_df.empty else '<p>No sport-specific data available</p>'}
        
        <h4>Kelly Criterion Validation</h4>
        <p>Actual win rate vs expected provides insight into edge estimation accuracy.</p>
        <p>Difference: {(win_rate - expected_win_rate):.2%} (positive = better than expected)</p>
        """
        
        self.report_sections.append(report)
    
    def analyze_correlation_and_bias(self):
        """Analyze correlation between Kalshi and sportsbook probabilities, detect bias"""
        if self.odds_data.empty:
            self.report_sections.append("<h3>📈 Correlation Analysis</h3><p>No odds data available</p>")
            return
        
        valid_data = self.odds_data[
            (self.odds_data['kalshi_implied_odds'] > 0) & 
            (self.odds_data['kalshi_implied_odds'] < 1) &
            (self.odds_data['composite_devigged_odds'] > 0) & 
            (self.odds_data['composite_devigged_odds'] < 1)
        ].copy()
        
        if len(valid_data) < 10:
            self.report_sections.append("<h3>📈 Correlation Analysis</h3><p>Insufficient valid data for correlation analysis</p>")
            return
        
        kalshi_probs = valid_data['kalshi_implied_odds'].values
        sportsbook_probs = valid_data['composite_devigged_odds'].values
        
        correlation = np.corrcoef(kalshi_probs, sportsbook_probs)[0, 1]
        
        kalshi_mean = kalshi_probs.mean()
        sportsbook_mean = sportsbook_probs.mean()
        
        kalshi_extreme = np.mean(np.abs(kalshi_probs - 0.5))
        sportsbook_extreme = np.mean(np.abs(sportsbook_probs - 0.5))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        merged_data = self.merge_data_with_results()
        if not merged_data.empty:
            scatter_data = valid_data.merge(
                merged_data[['timestamp', 'team', 'team_won']], 
                on=['timestamp', 'team'], 
                how='left'
            )
            
            won_mask = scatter_data['team_won'] == 1
            lost_mask = scatter_data['team_won'] == 0
            unknown_mask = scatter_data['team_won'].isna()
            
            if won_mask.any():
                ax.scatter(sportsbook_probs[won_mask], kalshi_probs[won_mask], 
                          c='green', alpha=0.6, label='Won', s=20)
            if lost_mask.any():
                ax.scatter(sportsbook_probs[lost_mask], kalshi_probs[lost_mask], 
                          c='red', alpha=0.6, label='Lost', s=20)
            if unknown_mask.any():
                ax.scatter(sportsbook_probs[unknown_mask], kalshi_probs[unknown_mask], 
                          c='gray', alpha=0.3, label='Unknown', s=20)
            ax.legend()
        else:
            ax.scatter(sportsbook_probs, kalshi_probs, alpha=0.6, s=20)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Correlation')
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(sportsbook_probs, kalshi_probs)
        line_x = np.array([0, 1])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r-', alpha=0.8, label=f'Regression (R²={r_value**2:.3f})')
        
        ax.set_xlabel('Sportsbook Probability')
        ax.set_ylabel('Kalshi Probability')
        ax.set_title(f'Kalshi vs Sportsbook Probabilities (r={correlation:.3f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        correlation_chart_path = save_chart_for_email(fig, "correlation_analysis.png")
        self.charts.append(correlation_chart_path)
        
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            ts_data = valid_data.sort_values('timestamp')[['kalshi_implied_odds', 'composite_devigged_odds']].values
            
            if len(ts_data) > 20:
                gc_result = grangercausalitytests(ts_data, maxlag=3, verbose=False)
                granger_p = min([gc_result[lag][0]['ssr_ftest'][1] for lag in gc_result.keys()])
                granger_text = f"Granger causality p-value: {granger_p:.4f} ({'significant' if granger_p < 0.05 else 'not significant'})"
            else:
                granger_text = "Insufficient data for Granger causality test"
        except ImportError:
            granger_text = "Statsmodels not available for Granger causality test"
        except Exception as e:
            granger_text = f"Granger causality test failed: {str(e)}"
        
        report = f"""
        <h3>📈 Correlation & Bias Analysis</h3>
        <p><strong>Correlation:</strong> {correlation:.3f}</p>
        <p><strong>Regression:</strong> Kalshi = {slope:.3f} × Sportsbook + {intercept:.3f} (R² = {r_value**2:.3f})</p>
        
        <h4>Bias Detection</h4>
        <p><strong>Average Kalshi Probability:</strong> {kalshi_mean:.3f}</p>
        <p><strong>Average Sportsbook Probability:</strong> {sportsbook_mean:.3f}</p>
        <p><strong>Kalshi Extremeness:</strong> {kalshi_extreme:.3f} (distance from 0.5)</p>
        <p><strong>Sportsbook Extremeness:</strong> {sportsbook_extreme:.3f} (distance from 0.5)</p>
        
        <h4>Market Efficiency</h4>
        <p>{granger_text}</p>
        <p>Higher correlation suggests markets are efficient and prices converge quickly.</p>
        """
        
        self.report_sections.append(report)
    
    def generate_html_report(self) -> str:
        """Generate complete HTML report"""
        today = datetime.now(eastern).strftime("%Y-%m-%d")
        
        html_report = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h3 {{ color: #34495e; margin-top: 30px; }}
                h4 {{ color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .metric {{ background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Weekly Diagnostics Report - {today}</h1>
                <p><em>Comprehensive analysis of trading strategy performance and market efficiency</em></p>
                
                {''.join(self.report_sections)}
                
                <h3>📋 Summary & Recommendations</h3>
                <p>This report analyzes all available historical data to identify strategy improvements.</p>
                <p><strong>Data Points Analyzed:</strong></p>
                <ul>
                    <li>Odds Records: {len(self.odds_data)}</li>
                    <li>Placed Orders: {len(self.orders_data)}</li>
                    <li>Game Results: {len(self.results_data)}</li>
                </ul>
                
                <p><em>Report generated on {datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')}</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_report
    
    def save_and_send_report(self):
        """Save report locally and send via email if configured"""
        html_report = self.generate_html_report()
        
        report_filename = f"diagnostics_report_{datetime.now(eastern).strftime('%Y%m%d')}.html"
        with open(report_filename, 'w') as f:
            f.write(html_report)
        print(f"✅ Report saved to {report_filename}")
        
        if EMAIL_ENABLED:
            try:
                msg = MIMEMultipart()
                msg['Subject'] = f"📊 Weekly Diagnostics Report - {datetime.now(eastern).strftime('%Y-%m-%d')}"
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
                print("✅ Email sent successfully!")
                
            except Exception as e:
                print(f"❌ Failed to send email: {e}")
        else:
            print("📧 Email not configured - report saved locally only")
    
    def run_full_analysis(self):
        """Run complete diagnostics analysis"""
        print("🔍 Starting weekly diagnostics analysis...")
        
        print("📊 Analyzing accuracy: Kalshi vs Sportsbooks...")
        self.analyze_accuracy_kalshi_vs_sportsbooks()
        
        print("🔍 Detecting arbitrage opportunities...")
        self.analyze_arbitrage_opportunities()
        
        print("💰 Analyzing profit performance...")
        self.analyze_profit_performance()
        
        print("📈 Analyzing correlations and bias...")
        self.analyze_correlation_and_bias()
        
        print("📧 Generating and sending report...")
        self.save_and_send_report()
        
        print("✅ Diagnostics analysis complete!")

def main():
    analyzer = DiagnosticsAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
