import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import pytz
from typing import Dict, List, Tuple, Optional
import requests
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

eastern = pytz.timezone("US/Eastern")

def load_odds_timeseries_data(start_date: Optional[str] = None, end_date: Optional[str] = None, sport: Optional[str] = None) -> pd.DataFrame:
    """Load odds timeseries data from NDJSON files or legacy JSON"""
    data = []
    
    odds_dir = "odds_timeseries"
    if os.path.exists(odds_dir):
        for sport_dir in os.listdir(odds_dir):
            sport_path = os.path.join(odds_dir, sport_dir)
            if not os.path.isdir(sport_path):
                continue
            
            if sport and sport.upper() != sport_dir.upper():
                continue
                
            for filename in os.listdir(sport_path):
                if not filename.endswith('.ndjson'):
                    continue
                    
                file_date = filename.replace('.ndjson', '')
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                    
                filepath = os.path.join(sport_path, filename)
                try:
                    df = pd.read_json(filepath, lines=True)
                    data.append(df)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    if not data:
        legacy_path = "odds_timeseries.json"
        if os.path.exists(legacy_path):
            try:
                with open(legacy_path, 'r') as f:
                    legacy_data = json.load(f)
                df = pd.DataFrame(legacy_data)
                if not df.empty:
                    data.append(df)
            except Exception as e:
                print(f"Error reading legacy odds data: {e}")
    
    if not data:
        return pd.DataFrame()
    
    combined_df = pd.concat(data, ignore_index=True)
    
    if start_date or end_date:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        if start_date:
            start_datetime = pd.to_datetime(start_date).tz_localize('UTC').tz_convert(combined_df['timestamp'].dt.tz)
            combined_df = combined_df[combined_df['timestamp'] >= start_datetime]
        if end_date:
            end_datetime = pd.to_datetime(end_date).tz_localize('UTC').tz_convert(combined_df['timestamp'].dt.tz) + pd.Timedelta(days=1)
            combined_df = combined_df[combined_df['timestamp'] < end_datetime]
    
    return combined_df

def load_placed_orders() -> pd.DataFrame:
    """Load placed orders data"""
    orders_path = "placed_orders.json"
    if not os.path.exists(orders_path):
        return pd.DataFrame()
    
    try:
        with open(orders_path, 'r') as f:
            orders_data = json.load(f)
        df = pd.DataFrame(orders_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading placed orders: {e}")
        return pd.DataFrame()

def load_game_results() -> pd.DataFrame:
    """Load game results data"""
    results_path = "game_results.json"
    if not os.path.exists(results_path):
        return pd.DataFrame()
    
    try:
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        flattened = []
        for entry in results_data:
            date = entry.get('date')
            for game in entry.get('games', []):
                game['date'] = date
                flattened.append(game)
        
        return pd.DataFrame(flattened)
    except Exception as e:
        print(f"Error loading game results: {e}")
        return pd.DataFrame()

def calculate_brier_score(predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
    """Calculate Brier score (lower is better)"""
    return np.mean((predicted_probs - actual_outcomes) ** 2)

def calculate_log_loss(predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
    """Calculate log loss (lower is better)"""
    predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)
    return -np.mean(actual_outcomes * np.log(predicted_probs) + 
                   (1 - actual_outcomes) * np.log(1 - predicted_probs))

def calculate_calibration(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate calibration curve"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = actual_outcomes[in_bin].mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            
            bin_centers.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)

def detect_arbitrage_opportunities(odds_data: pd.DataFrame) -> pd.DataFrame:
    """Detect arbitrage opportunities in odds data"""
    arb_opportunities = []
    
    for _, row in odds_data.iterrows():
        per_book_odds = row.get('per_book_american_odds', {})
        if not per_book_odds or len(per_book_odds) < 2:
            continue
        
        implied_probs = {}
        for book, odds in per_book_odds.items():
            if odds > 0:
                implied_prob = 100 / (odds + 100)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100)
            implied_probs[book] = implied_prob
        
        books = list(implied_probs.keys())
        for i in range(len(books)):
            for j in range(i + 1, len(books)):
                book1, book2 = books[i], books[j]
                
                total_inverse_prob = (1 / implied_probs[book1]) + (1 / implied_probs[book2])
                if total_inverse_prob < 1.0:  # Arbitrage opportunity
                    profit_margin = 1.0 - total_inverse_prob
                    arb_opportunities.append({
                        'timestamp': row['timestamp'],
                        'sport': row['sport'],
                        'team': row['team'],
                        'book1': book1,
                        'book2': book2,
                        'book1_prob': implied_probs[book1],
                        'book2_prob': implied_probs[book2],
                        'profit_margin': profit_margin
                    })
        
        kalshi_prob = row.get('kalshi_implied_odds', 0)
        if kalshi_prob > 0:
            for book, book_prob in implied_probs.items():
                total_inverse = (1 / kalshi_prob) + (1 / (1 - book_prob))
                if total_inverse < 1.0:
                    profit_margin = 1.0 - total_inverse
                    arb_opportunities.append({
                        'timestamp': row['timestamp'],
                        'sport': row['sport'],
                        'team': row['team'],
                        'book1': 'Kalshi',
                        'book2': book,
                        'book1_prob': kalshi_prob,
                        'book2_prob': 1 - book_prob,
                        'profit_margin': profit_margin
                    })
    
    return pd.DataFrame(arb_opportunities)

def create_calibration_plot(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, title: str = "Calibration Plot") -> plt.Figure:
    """Create calibration plot"""
    bin_centers, bin_accuracies, bin_counts = calculate_calibration(predicted_probs, actual_outcomes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(bin_centers, bin_accuracies, 'o-', label='Calibration', linewidth=2, markersize=8)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
    
    for i, (x, y, count) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
        if count > 0:
            ax.annotate(f'n={count}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

def save_chart_for_email(fig: plt.Figure, filename: str) -> str:
    """Save chart and return path for email attachment"""
    filepath = f"/tmp/{filename}"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath
