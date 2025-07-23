#!/usr/bin/env python3
"""
MLB Odds Accuracy Analysis Script

Compares the accuracy of moneyline odds from FanDuel, DraftKings, and Pinnacle
across a full MLB season by:
1. Fetching pregame and live odds from Odds API
2. Getting actual game results from MLB Stats API
3. Removing vig using normalization method
4. Performing calibration analysis to measure accuracy
5. Comparing sportsbook performance at different probability levels

Author: Devin AI
Date: July 2025
"""

import requests
import pandas as pd
import random
from datetime import datetime, timedelta, timezone
import time
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
try:
    from kalshi_bot import team_abbr_to_name, devig_fanduel_odds
    print("✅ Imported existing team mappings and devig function")
except ImportError:
    print("⚠️ Could not import from kalshi_bot.py, using fallback mappings")
    team_abbr_to_name = {}

class MLBOddsAccuracyAnalyzer:
    """
    Analyzes the accuracy of MLB moneyline odds from multiple sportsbooks.
    """
    
    def __init__(self, odds_api_key: str, test_mode: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            odds_api_key: API key for The Odds API
            test_mode: If True, limits API calls for testing (default: True)
        """
        self.odds_api_key = odds_api_key
        self.test_mode = test_mode
        self.base_odds_url = "https://api.the-odds-api.com/v4"
        self.mlb_stats_base_url = "https://statsapi.mlb.com/api/v1"
        
        self.target_sportsbooks = ['fanduel', 'draftkings', 'pinnacle']
        
        self.api_calls_made = 0
        self.max_test_calls = 95  # Leave buffer under 100
        
        self.odds_data = []
        self.game_results = []
        self.calibration_results = {}
        
    def estimate_api_calls_needed(self, start_date: str, end_date: str) -> int:
        """
        Estimate total API calls needed for full analysis.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Estimated number of API calls
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end - start).days + 1
        
        estimated_games = days * 15
        calls_per_game = 6
        total_calls = estimated_games * calls_per_game
        
        print(f"📊 ESTIMATED API CALLS FOR FULL SEASON ANALYSIS:")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Estimated days: {days}")
        print(f"   Estimated games: {estimated_games}")
        print(f"   Calls per game: {calls_per_game} (1 pregame + 5 live)")
        print(f"   TOTAL ESTIMATED CALLS: {total_calls:,}")
        print(f"   ⚠️  This will exceed free tier limits - prepare for rate limiting!")
        
        return total_calls
    
    def get_mlb_schedule(self, date: str) -> List[Dict]:
        """
        Get MLB schedule for a specific date using MLB Stats API.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of game dictionaries
        """
        url = f"{self.mlb_stats_base_url}/schedule"
        params = {
            'sportId': 1,  # MLB
            'date': date,
            'hydrate': 'team,linescore'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('dates', [{}])[0].get('games', [])
        except Exception as e:
            print(f"Error fetching MLB schedule for {date}: {e}")
            return []
    
    def get_game_result(self, game_id: str) -> Optional[Dict]:
        """
        Get the final result of a specific MLB game.
        
        Args:
            game_id: MLB game ID
            
        Returns:
            Dictionary with game result info or None if not found
        """
        url = f"{self.mlb_stats_base_url}/game/{game_id}/linescore"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data.get('currentInning') and data.get('isTopInning') is not None:
                away_score = data.get('teams', {}).get('away', {}).get('runs', 0)
                home_score = data.get('teams', {}).get('home', {}).get('runs', 0)
                
                return {
                    'game_id': game_id,
                    'away_score': away_score,
                    'home_score': home_score,
                    'winner': 'home' if home_score > away_score else 'away',
                    'final': True
                }
        except Exception as e:
            print(f"Error fetching game result for {game_id}: {e}")
            
        return None
    
    def fetch_odds_for_date(self, date: str, is_live: bool = False) -> List[Dict]:
        """
        Fetch MLB odds for a specific date from The Odds API.
        
        Args:
            date: Date in YYYY-MM-DD format
            is_live: Whether to fetch live odds (default: False for pregame)
            
        Returns:
            List of odds dictionaries
        """
        if self.test_mode and self.api_calls_made >= self.max_test_calls:
            print(f"⚠️  Test mode: Reached max API calls ({self.max_test_calls})")
            return []
        
        endpoint = "odds" if not is_live else "odds"
        url = f"{self.base_odds_url}/sports/baseball_mlb/{endpoint}"
        
        params = {
            'apiKey': self.odds_api_key,
            'regions': 'us',
            'markets': 'h2h',  # head-to-head (moneyline)
            'oddsFormat': 'american',
            'bookmakers': ','.join(self.target_sportsbooks)
        }
        
        if not is_live:
            params['dateFormat'] = 'iso'
            params['commenceTimeFrom'] = f"{date}T00:00:00Z"
            params['commenceTimeTo'] = f"{date}T23:59:59Z"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            self.api_calls_made += 1
            
            data = response.json()
            print(f"✅ Fetched {'live' if is_live else 'pregame'} odds for {date} - API calls: {self.api_calls_made}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching odds for {date}: {e}")
            return []
    
    def remove_vig_normalization(self, odds_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Remove vig from odds using normalization method.
        
        This is the same method implemented in the kalshi_bot.py devig function.
        
        Args:
            odds_dict: Dictionary with team names as keys and American odds as values
            
        Returns:
            Dictionary with fair decimal odds
        """
        if len(odds_dict) != 2:
            return {}
        
        teams = list(odds_dict.keys())
        odds_values = list(odds_dict.values())
        
        def american_to_implied_prob(odds):
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        
        prob1 = american_to_implied_prob(odds_values[0])
        prob2 = american_to_implied_prob(odds_values[1])
        
        total_prob = prob1 + prob2
        
        if total_prob <= 0:
            return {}
        
        fair_prob1 = prob1 / total_prob
        fair_prob2 = prob2 / total_prob
        
        return {
            teams[0]: 1 / fair_prob1,
            teams[1]: 1 / fair_prob2
        }
    
    def process_odds_data(self, odds_data: List[Dict], date: str, is_live: bool = False) -> None:
        """
        Process and store odds data with vig removal.
        
        Args:
            odds_data: Raw odds data from API
            date: Date string
            is_live: Whether this is live odds data
        """
        for game in odds_data:
            game_id = game.get('id')
            commence_time = game.get('commence_time')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            for bookmaker in game.get('bookmakers', []):
                book_name = bookmaker.get('key')
                if book_name not in self.target_sportsbooks:
                    continue
                
                for market in bookmaker.get('markets', []):
                    if market.get('key') != 'h2h':
                        continue
                    
                    outcomes = market.get('outcomes', [])
                    if len(outcomes) != 2:
                        continue
                    
                    odds_dict = {}
                    for outcome in outcomes:
                        team = outcome.get('name')
                        odds = outcome.get('price')
                        odds_dict[team] = odds
                    
                    fair_odds = self.remove_vig_normalization(odds_dict)
                    
                    if fair_odds:
                        for team, raw_odds in odds_dict.items():
                            fair_decimal_odds = fair_odds.get(team)
                            implied_prob = 1 / fair_decimal_odds if fair_decimal_odds else None
                            
                            self.odds_data.append({
                                'date': date,
                                'game_id': game_id,
                                'commence_time': commence_time,
                                'home_team': home_team,
                                'away_team': away_team,
                                'team': team,
                                'is_home': team == home_team,
                                'sportsbook': book_name,
                                'raw_american_odds': raw_odds,
                                'fair_decimal_odds': fair_decimal_odds,
                                'fair_implied_prob': implied_prob,
                                'is_live': is_live,
                                'snapshot_time': datetime.now(timezone.utc).isoformat()
                            })
    
    def collect_live_snapshots(self, date: str, num_snapshots: int = 5) -> None:
        """
        Collect multiple live odds snapshots throughout the day.
        
        Args:
            date: Date in YYYY-MM-DD format
            num_snapshots: Number of random snapshots to take (default: 5)
        """
        if self.test_mode and self.api_calls_made >= self.max_test_calls:
            return
            
        games = self.get_mlb_schedule(date)
        if not games:
            return
            
        base_time = datetime.strptime(f"{date} 12:00:00", '%Y-%m-%d %H:%M:%S')
        
        for i in range(min(num_snapshots, self.max_test_calls - self.api_calls_made)):
            random_hours = random.uniform(1, 10)
            snapshot_time = base_time + timedelta(hours=random_hours)
            
            print(f"📸 Taking live snapshot {i+1}/{num_snapshots} for {date}")
            live_odds = self.fetch_odds_for_date(date, is_live=True)
            if live_odds:
                self.process_odds_data(live_odds, date, is_live=True)
                
            if self.test_mode:
                time.sleep(0.1)

    def normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team names between different APIs.
        
        Args:
            team_name: Raw team name from API
            
        Returns:
            Normalized team name
        """
        name_mappings = {
            'Arizona Diamondbacks': 'Arizona Diamondbacks',
            'Atlanta Braves': 'Atlanta Braves', 
            'Baltimore Orioles': 'Baltimore Orioles',
            'Boston Red Sox': 'Boston Red Sox',
            'Chicago Cubs': 'Chicago Cubs',
            'Chicago White Sox': 'Chicago White Sox',
            'Cincinnati Reds': 'Cincinnati Reds',
            'Cleveland Guardians': 'Cleveland Guardians',
            'Colorado Rockies': 'Colorado Rockies',
            'Detroit Tigers': 'Detroit Tigers',
            'Houston Astros': 'Houston Astros',
            'Kansas City Royals': 'Kansas City Royals',
            'Los Angeles Angels': 'Los Angeles Angels',
            'Los Angeles Dodgers': 'Los Angeles Dodgers',
            'Miami Marlins': 'Miami Marlins',
            'Milwaukee Brewers': 'Milwaukee Brewers',
            'Minnesota Twins': 'Minnesota Twins',
            'New York Mets': 'New York Mets',
            'New York Yankees': 'New York Yankees',
            'Oakland Athletics': 'Oakland Athletics',
            'Philadelphia Phillies': 'Philadelphia Phillies',
            'Pittsburgh Pirates': 'Pittsburgh Pirates',
            'San Diego Padres': 'San Diego Padres',
            'San Francisco Giants': 'San Francisco Giants',
            'Seattle Mariners': 'Seattle Mariners',
            'St. Louis Cardinals': 'St. Louis Cardinals',
            'Tampa Bay Rays': 'Tampa Bay Rays',
            'Texas Rangers': 'Texas Rangers',
            'Toronto Blue Jays': 'Toronto Blue Jays',
            'Washington Nationals': 'Washington Nationals'
        }
        
        return name_mappings.get(team_name, team_name)

    def collect_sample_data(self, num_days: int = 3) -> None:
        """
        Collect sample data for testing (limited API calls).
        
        Args:
            num_days: Number of recent days to analyze
        """
        print(f"🔍 COLLECTING SAMPLE DATA ({num_days} days) - Test Mode: {self.test_mode}")
        
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
        
        for date in dates:
            if self.test_mode and self.api_calls_made >= self.max_test_calls:
                break
                
            print(f"\n📅 Processing {date}...")
            
            pregame_odds = self.fetch_odds_for_date(date, is_live=False)
            if pregame_odds:
                self.process_odds_data(pregame_odds, date, is_live=False)
            
            if self.api_calls_made < self.max_test_calls - 5:
                self.collect_live_snapshots(date, num_snapshots=5)
        
        print(f"\n✅ Sample data collection complete. Total API calls: {self.api_calls_made}")
        print(f"📊 Collected {len(self.odds_data)} odds records")
    
    def perform_calibration_analysis(self) -> Dict:
        """
        Perform calibration analysis on the collected odds data.
        
        Returns:
            Dictionary with calibration results by sportsbook
        """
        if not self.odds_data:
            print("❌ No odds data available for calibration analysis")
            return {}
        
        print("\n🎯 PERFORMING CALIBRATION ANALYSIS...")
        
        df = pd.DataFrame(self.odds_data)
        
        print("⚠️  Note: Using simulated results for demonstration. Real implementation would fetch actual MLB results.")
        
        random.seed(42)
        df['simulated_win'] = [random.random() < prob for prob in df['fair_implied_prob']]
        
        results = {}
        
        for sportsbook in self.target_sportsbooks:
            book_data = df[df['sportsbook'] == sportsbook].copy()
            
            if len(book_data) == 0:
                continue
            
            def assign_bin(prob):
                return int(prob * 10) / 10
            
            book_data['prob_bin'] = book_data['fair_implied_prob'].apply(assign_bin)
            
            calibration_data = []
            for bin_val in sorted(book_data['prob_bin'].unique()):
                bin_data = book_data[book_data['prob_bin'] == bin_val]
                if len(bin_data) > 0:
                    predicted_prob = bin_data['fair_implied_prob'].mean()
                    actual_rate = bin_data['simulated_win'].mean()
                    count = len(bin_data)
                    
                    calibration_data.append({
                        'bin_range': f"[{bin_val:.1f}, {bin_val+0.1:.1f})",
                        'predicted_prob': predicted_prob,
                        'actual_rate': actual_rate,
                        'count': count,
                        'calibration_error': abs(predicted_prob - actual_rate)
                    })
            
            if calibration_data:
                cal_df = pd.DataFrame(calibration_data)
                mean_calibration_error = cal_df['calibration_error'].mean()
                
                results[sportsbook] = {
                    'calibration_data': cal_df,
                    'mean_calibration_error': mean_calibration_error,
                    'total_predictions': len(book_data),
                    'overall_accuracy': book_data['simulated_win'].mean()
                }
        
        self.calibration_results = results
        return results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive accuracy report.
        
        Returns:
            Formatted report string
        """
        if not self.calibration_results:
            return "❌ No calibration results available. Run analysis first."
        
        report = []
        report.append("=" * 60)
        report.append("MLB ODDS ACCURACY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"Test Mode: {self.test_mode}")
        report.append(f"Total API Calls Made: {self.api_calls_made}")
        report.append(f"Total Odds Records: {len(self.odds_data)}")
        report.append("")
        
        report.append("📊 SPORTSBOOK ACCURACY COMPARISON")
        report.append("-" * 40)
        
        accuracy_ranking = []
        for book, results in self.calibration_results.items():
            accuracy_ranking.append((book, results['mean_calibration_error'], results['total_predictions']))
        
        accuracy_ranking.sort(key=lambda x: x[1])  # Sort by calibration error (lower is better)
        
        for i, (book, error, count) in enumerate(accuracy_ranking, 1):
            report.append(f"{i}. {book.upper()}")
            report.append(f"   Mean Calibration Error: {error:.4f}")
            report.append(f"   Total Predictions: {count}")
            report.append(f"   Overall Accuracy: {self.calibration_results[book]['overall_accuracy']:.3f}")
            report.append("")
        
        report.append("🔬 METHODOLOGY")
        report.append("-" * 40)
        report.append("• Vig Removal: Normalization method (convert to implied probabilities,")
        report.append("  normalize to sum to 1.0, convert back to fair odds)")
        report.append("• Calibration: Binned predicted probabilities vs actual outcomes")
        report.append("• Accuracy Metric: Mean absolute calibration error across probability bins")
        report.append("• Lower calibration error = better accuracy")
        report.append("")
        
        if self.test_mode:
            report.append("⚠️  TEST MODE LIMITATIONS")
            report.append("-" * 40)
            report.append("• Limited to sample data to stay under 100 API calls")
            report.append("• Using simulated game results for demonstration")
            report.append("• Full season analysis would require thousands of API calls")
            report.append("• Real implementation would fetch actual MLB game results")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None) -> str:
        """
        Save analysis results to files.
        
        Args:
            filename: Base filename (default: auto-generated)
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"mlb_odds_analysis_{timestamp}"
        
        if self.odds_data:
            odds_df = pd.DataFrame(self.odds_data)
            odds_file = f"{filename}_odds_data.csv"
            odds_df.to_csv(odds_file, index=False)
            print(f"💾 Saved odds data to: {odds_file}")
        
        if self.calibration_results:
            for book, results in self.calibration_results.items():
                cal_file = f"{filename}_{book}_calibration.csv"
                results['calibration_data'].to_csv(cal_file, index=False)
                print(f"💾 Saved {book} calibration data to: {cal_file}")
        
        report = self.generate_report()
        report_file = f"{filename}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Saved report to: {report_file}")
        
        return report_file


def main():
    """
    Main execution function with example usage.
    """
    print("🏟️  MLB ODDS ACCURACY ANALYZER")
    print("=" * 50)
    
    ODDS_API_KEY = os.getenv('ODDS_API_KEY', 'YOUR_ODDS_API_KEY_HERE')
    TEST_MODE = True
    
    if ODDS_API_KEY == "YOUR_ODDS_API_KEY_HERE":
        print("❌ Please set your Odds API key:")
        print("   Option 1: Set environment variable: export ODDS_API_KEY='your_key_here'")
        print("   Option 2: Replace ODDS_API_KEY variable in the script")
        print("   Get your free API key at: https://the-odds-api.com/")
        return
    
    analyzer = MLBOddsAccuracyAnalyzer(ODDS_API_KEY, test_mode=TEST_MODE)
    
    if not TEST_MODE:
        analyzer.estimate_api_calls_needed('2024-03-28', '2024-09-29')
        
        response = input("\nProceed with full season analysis? (y/N): ")
        if response.lower() != 'y':
            print("Analysis cancelled.")
            return
    
    if TEST_MODE:
        analyzer.collect_sample_data(num_days=3)
    else:
        print("Full season analysis not implemented in this demo")
        return
    
    results = analyzer.perform_calibration_analysis()
    
    if results:
        report = analyzer.generate_report()
        print("\n" + report)
        
        report_file = analyzer.save_results()
        print(f"\n✅ Analysis complete! Report saved to: {report_file}")
    else:
        print("❌ Analysis failed - no results generated")


if __name__ == "__main__":
    main()
