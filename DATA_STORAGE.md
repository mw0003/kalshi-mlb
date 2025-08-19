Odds and Orders Data Storage

odds_timeseries.json (ODDS_TIMESERIES_PATH)
- One entry per team per run.
- Retention: indefinite.

Fields per entry:
- timestamp: ISO8601 string (US/Eastern)
- sport: Uppercase sport key (e.g., MLB, NFL, WNBA, MLS, EPL)
- team: Team Name (matches Kalshi-mapped and sportsbook names)
- kalshi_implied_odds: Float, implied probability from Kalshi price (price_cents / 100)
- composite_devigged_odds: Float, composite probability after devig across books (1 / Composite Fair Odds)
- expected_value: Float, numeric edge used internally
- per_book_american_odds: Object mapping {book: american_price} (pre-devig raw feed)
- per_book_implied_prob: Object mapping {book: implied_probability_from_american_price} (pre-devig)
- composite_source_books: Array of books present for this team snapshot

Notes:
- Books currently captured: fanduel, draftkings, betmgm, caesars, espnbet (subject to availability per game).
- Historical retention is unlimited; consider rotating to NDJSON or per-day files if size becomes large.

placed_orders.json (PLACED_ORDERS_PATH)
- Appends an entry per placed order.

Fields per entry:
- timestamp: ISO8601 string (US/Eastern)
- team: Team Name
- ticker: Kalshi market ticker
- contracts: Integer count
- price: Entered price in cents
- total_cost: Dollar cost of the position
- expected_value_before_devig: Percentage in basis points terms relative to 100 (derived)
- expected_value_after_devig: Percentage edge after devig (as percent number)
