Odds and Orders Data Storage

NDJSON daily timeseries (recommended for analysis)
- Path pattern: odds_timeseries/{SPORT}/YYYY-MM-DD.ndjson
- One JSON object per line (newline-delimited JSON)
- Append-only; one file per sport per day
- Retention: indefinite (no pruning)

Fields per record:
- timestamp: ISO8601 string (US/Eastern)
- sport: Uppercase sport key (e.g., MLB, NFL, WNBA, MLS, EPL)
- team: Team Name (matches Kalshi-mapped and sportsbook names)
- kalshi_implied_odds: Float, implied probability from Kalshi price (price_cents / 100)
- composite_devigged_odds: Float, composite probability after devig across books (1 / Composite Fair Odds)
- expected_value: Float, numeric edge used internally
- per_book_american_odds: Object mapping {book: american_price} (pre-devig raw feed)
- per_book_implied_prob: Object mapping {book: implied_probability_from_american_price} (pre-devig)
- composite_source_books: Array of books present for this team snapshot

Reading examples:
- jq: jq -c . odds_timeseries/MLB/2025-08-19.ndjson | head
- pandas: import pandas as pd; pd.read_json("odds_timeseries/MLB/2025-08-19.ndjson", lines=True)

Legacy JSON array (backward compatibility)
- File: odds_timeseries.json (ODDS_TIMESERIES_PATH)
- Appends entries in-memory then writes full JSON array
- Retention: indefinite
- Disable legacy write by setting environment variable NDJSON_ONLY=1

Notes:
- Books currently captured: fanduel, draftkings, betmgm, caesars, espnbet (subject to availability per game).
- Because NDJSON is per-sport and per-day, files remain manageable and easy to query or archive.

placed_orders.json (PLACED_ORDERS_PATH)
- Appends an entry per placed order

Fields per entry:
- timestamp: ISO8601 string (US/Eastern)
- team: Team Name
- ticker: Kalshi market ticker
- contracts: Integer count
- price: Entered price in cents
- total_cost: Dollar cost of the position
- expected_value_before_devig: Percentage in basis points terms relative to 100 (derived)
- expected_value_after_devig: Percentage edge after devig (as percent number)
