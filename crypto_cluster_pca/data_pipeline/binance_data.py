#!/usr/bin/env python3
"""
Binance daily OHLCV loader for the regime-research pipeline.

Replaces the CoinGecko collector in backtest_system/backtest_data_manager.py, whose
`days=min(n, 365)` request returned a *trailing* window anchored on the request date
and silently ignored the caller's start/end dates (see Overfitting_Diagnosis.md, item 3).

Two properties this loader guarantees and the old one did not:

1. Full history, not a trailing slice. Binance caps klines at 1000 per call, so the
   fetch paginates on `since` until the exchange stops advancing. A single unpaginated
   call reproduces exactly the bug being corrected.
2. The cache filename is derived from the data actually returned, never from the
   requested range, so a filename can no longer assert a window the contents lack.

Network note: api.binance.com returns HTTP 451 from restricted locations. The public
market-data mirror data-api.binance.vision serves the identical global-Binance klines
without that restriction, so all REST hosts are routed there.
"""

import json
import time
from pathlib import Path

import ccxt
import pandas as pd

VISION_HOST = "https://data-api.binance.vision/api/v3"

# The 15-asset universe the project already analyzes
# (backtest_data_manager.py:31-47), mapped to Binance USDT spot pairs.
# Stablecoins are excluded — see STABLECOINS_EXCLUDED below.
UNIVERSE = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "ADA/USDT": "cardano",
    "DOT/USDT": "polkadot",
    "LINK/USDT": "chainlink",
    "SOL/USDT": "solana",
    "MATIC/USDT": "matic-network",
    "AVAX/USDT": "avalanche-2",
    "ATOM/USDT": "cosmos",
    "ALGO/USDT": "algorand",
    "XRP/USDT": "ripple",
    "BNB/USDT": "binancecoin",
    "DOGE/USDT": "dogecoin",
    "LTC/USDT": "litecoin",
}

# tether (USDT) is the quote currency — there is no USDT/USDT pair. It is also a
# dollar peg: near-zero return variance would dominate StandardScaler and load onto a
# PCA component as pure noise. Dropped, which also resolves Overfitting_Diagnosis.md
# finding 5.6 (USDT was scored and tradeable in the old strategy).
STABLECOINS_EXCLUDED = {"tether": "USDT is the quote currency and a dollar peg"}

CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache" / "binance_daily"
HISTORY_START = "2017-01-01T00:00:00Z"


def make_exchange() -> ccxt.binance:
    """Binance client routed at the unrestricted public market-data mirror."""
    ex = ccxt.binance({"enableRateLimit": True, "timeout": 30000})
    ex.urls["api"] = {key: VISION_HOST for key in ex.urls["api"]}
    return ex


def fetch_full_daily(ex, symbol: str, start_iso: str = HISTORY_START) -> pd.DataFrame:
    """
    Page through daily klines from `start_iso` to the present.

    Binance returns at most 1000 candles per call. Without this loop the result is a
    trailing window regardless of `since` — the failure mode being corrected here.
    """
    since = ex.parse8601(start_iso)
    day_ms = 24 * 60 * 60 * 1000
    rows: list = []

    while True:
        batch = ex.fetch_ohlcv(symbol, "1d", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + day_ms
        if nxt <= since:  # exchange stopped advancing; avoid an infinite loop
            break
        since = nxt
        if len(batch) < 1000:  # reached the present
            break
        time.sleep(ex.rateLimit / 1000)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()

    # The current UTC day is a partial candle; including it injects a fractional-day
    # return into every downstream feature.
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    df = df[df.index < today]

    return df.astype(float)


def collect(symbols: dict = None, force_refresh: bool = False) -> dict:
    """Fetch (or load) daily OHLCV for the universe. Returns {symbol: DataFrame}."""
    symbols = symbols or UNIVERSE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ex = make_exchange()
    ex.load_markets()

    out, missing = {}, []
    for sym in symbols:
        slug = sym.replace("/", "")
        cache = CACHE_DIR / f"{slug}_1d.parquet"
        if cache.exists() and not force_refresh:
            out[sym] = pd.read_parquet(cache)
            continue
        if sym not in ex.markets:
            missing.append(sym)
            continue
        df = fetch_full_daily(ex, sym)
        if df.empty:
            missing.append(sym)
            continue
        df.to_parquet(cache)
        out[sym] = df
        print(f"  {sym:12s} {len(df):5d} candles  {df.index[0].date()} -> {df.index[-1].date()}")

    if missing:
        print(f"  no daily history available: {missing}")

    manifest = {
        "fetched_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "source": VISION_HOST,
        "interval": "1d",
        "stablecoins_excluded": STABLECOINS_EXCLUDED,
        "symbols": {
            s: {
                "rows": len(d),
                "first": str(d.index[0].date()),
                "last": str(d.index[-1].date()),
            }
            for s, d in out.items()
        },
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


if __name__ == "__main__":
    print("Fetching Binance daily history...")
    data = collect()
    print(f"\n{len(data)} symbols cached to {CACHE_DIR}")


def load_stitched(force_refresh: bool = False) -> dict:
    """
    Universe data with the MATIC->POL migration stitched.

    Polygon rebranded MATIC to POL; Binance ended MATIC/USDT on 2024-09-10 and opened
    POL/USDT on 2024-09-13. The swap was 1:1 (observed seam ratio 1.083 across the
    3-day gap, i.e. ordinary drift, not a split), so the two series are concatenated
    with no rescaling. The 2-day gap is left as missing rather than filled.
    """
    data = collect(force_refresh=force_refresh)
    pol_cache = CACHE_DIR / "POLUSDT_1d.parquet"
    if pol_cache.exists() and not force_refresh:
        pol = pd.read_parquet(pol_cache)
    else:
        ex = make_exchange()
        ex.load_markets()
        pol = fetch_full_daily(ex, "POL/USDT")
        if not pol.empty:
            pol.to_parquet(pol_cache)
    if "MATIC/USDT" in data and not pol.empty:
        data["MATIC/USDT"] = pd.concat([data["MATIC/USDT"], pol]).sort_index()
    return data
