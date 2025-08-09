# Kalshi MLB Bot Edge Calculation Process

## Overview
This document explains how the Kalshi MLB betting bot calculates edge and makes betting decisions. The bot uses a sophisticated multi-step process to identify profitable betting opportunities by comparing Kalshi's implied odds to "fair" odds derived from multiple sportsbooks.

## Edge Calculation Pipeline

### 1. Odds Collection
- **Source**: Multiple sportsbooks via The Odds API (FanDuel, Pinnacle, DraftKings)
- **Format**: American odds (e.g., -110, +150)
- **Purpose**: Get market consensus on true probabilities

### 2. Devigging Process (`devig_sportsbook_odds`)
**Problem**: Sportsbooks add "vig" (vigorish) to ensure profit, making total implied probabilities > 100%

**Solution**: Probability normalization
```
Example:
Team A: -110 odds → 52.38% implied probability
Team B: -110 odds → 52.38% implied probability
Total: 104.76% (4.76% vig)

After devigging:
Team A: 52.38% / 104.76% = 50% fair probability
Team B: 52.38% / 104.76% = 50% fair probability
Total: 100% (vig removed)
```

### 3. Composite Odds Creation (`devig_composite_odds`)
- **Method**: Simple average of devigged probabilities across all available sportsbooks
- **Purpose**: Create more accurate "fair" odds by combining multiple market opinions
- **Output**: Composite fair odds for each team

### 4. Raw Edge Calculation (`fetch_sport_opportunities`)
**Formula**: `Edge = (Kalshi Decimal Odds × True Probability) - 1 - Fee`

**Components**:
- **Kalshi Decimal Odds**: 1 / (Kalshi Price in cents / 100)
- **True Probability**: 1 / Composite Fair Odds
- **Fee**: Variable based on Kalshi price (1.6% - 1.7% typically)

**Example**:
```
Kalshi Price: 60¢ → Decimal Odds: 1.67
Composite Fair Odds: 1.5 → True Probability: 66.67%
Expected Return: 1.67 × 0.6667 = 1.11
Raw Edge: 1.11 - 1 = 11%
Fee: 1.6%
Final Edge: 11% - 1.6% = 9.4%
```

### 5. Kelly Criterion Position Sizing (`kelly_wager`)
**Formula**: `Kelly Fraction = (bp - q) / b`
Where:
- b = odds received - 1 (Kalshi decimal odds - 1)
- p = true probability of winning
- q = probability of losing (1 - p)

**Implementation**:
```
edge = (fair_prob × (your_odds - 1)) - (1 - fair_prob)
kelly_fraction = edge / (your_odds - 1)
kelly_amount = kelly_fraction × bankroll
```

## Betting Decision Process

### 1. Market Filtering
**Criteria**:
- Kalshi price: 35¢ - 90¢ (avoids extreme probabilities)
- Edge: 3% - 15% (minimum profitability, maximum to avoid false positives)

### 2. Duplicate Prevention
- Checks today's executed orders
- Prevents betting on same team twice
- Prevents betting on both teams in same game

### 3. Position Sizing Safeguards
- **Kelly Multiplier**: 0.75 (reduces Kelly bet size for safety)
- **Maximum Position**: 20% of bankroll per bet
- **Maximum Wager Cap**: 30% of bankroll (additional safety)

### 4. Order Execution
- **Testing Mode**: Currently enabled (no real orders placed)
- **Order Type**: Limit orders at current ask price
- **Contract Calculation**: Wager amount ÷ cost per contract

## Risk Management Features

### 1. Conservative Kelly Implementation
- Uses 0.75 multiplier instead of full Kelly
- Caps individual positions at 20% of bankroll
- Additional 30% wager cap as backup safety

### 2. Price Range Filtering
- Avoids very low probability bets (< 35¢)
- Avoids very high probability bets (> 90¢)
- Focuses on "sweet spot" where edge detection is most reliable

### 3. Edge Range Filtering
- Minimum 3% edge ensures meaningful profit potential
- Maximum 15% edge filters out likely false positives
- Balances opportunity identification with false positive reduction

## Data Flow Summary

1. **Fetch Kalshi Markets** → Get available betting opportunities
2. **Fetch Sportsbook Odds** → Get market consensus from multiple books
3. **Devig Individual Books** → Remove vig from each sportsbook
4. **Create Composite Odds** → Average devigged odds across books
5. **Calculate Edge** → Compare Kalshi odds to composite fair odds
6. **Apply Filters** → Remove unsuitable opportunities
7. **Calculate Kelly Wagers** → Determine optimal bet sizes
8. **Check Duplicates** → Avoid double betting
9. **Place Orders** → Execute profitable bets (if not in testing mode)

## Key Insights

### Why This Approach Works
- **Market Inefficiency**: Kalshi prices may not perfectly reflect true probabilities
- **Information Aggregation**: Multiple sportsbooks provide better probability estimates
- **Mathematical Edge**: Systematic approach to identifying +EV opportunities

### Potential Limitations
- **Correlated Markets**: Sportsbooks may have similar biases
- **Limited Liquidity**: Kalshi markets may have wide spreads
- **Model Risk**: Composite odds may not represent true probabilities
- **Execution Risk**: Prices may move between calculation and order placement
