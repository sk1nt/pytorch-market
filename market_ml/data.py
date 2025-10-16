import pandas as pd
def load_sierra_chart_depth_bin(path: str) -> pd.DataFrame:
    """
    Load Sierra Chart .depth binary file. Returns DataFrame with timestamp, price, size, side, level.
    Format based on Reddit guide: 64-byte header, records with fields:
        DateTime (8 bytes), Command (1), Flags (1), NumOrders (2), price_raw (4), Quantity (4), Reserved (4)
    """
    import struct
    HEADER_SIZE = 64
    RECORD_SIZE = 24
    columns = ['timestamp', 'price', 'size', 'side', 'level', 'num_orders', 'flags', 'command']
    records = []
    with open(path, 'rb') as f:
        header = f.read(HEADER_SIZE)
        # Validate header (skip for now, could check FileTypeUniqueHeaderID)
        while True:
            rec_bytes = f.read(RECORD_SIZE)
            if len(rec_bytes) < RECORD_SIZE:
                break
            dt, cmd, flags, num_orders, price_raw, qty, reserved = struct.unpack('<QBBHfII', rec_bytes)
            ts = pd.to_datetime(dt, unit='ms')
            # Side/level logic: Command/Flags may encode side/level, but not documented. Store raw for now.
            records.append([ts, price_raw, qty, flags, cmd, num_orders, flags, cmd])
    df = pd.DataFrame(records, columns=columns)
    return df
def stream_sierra_chart_scid(path: str, poll_interval: float = 0.5, start_record: int = 0):
    """
    Generator that yields new records from a .scid file as they appear (real-time scanning).
    Args:
        path: Path to .scid file
        poll_interval: Seconds to wait between polling for new records
        start_record: Record index to start from (default 0)
    Yields:
        dict with keys: timestamp, price, volume, num_trades, bid_volume, ask_volume
    """
    import time
    HEADER_SIZE = 56
    RECORD_SIZE = 40
    columns = [
        'timestamp', 'open', 'high', 'low', 'close',
        'num_trades', 'total_volume', 'bid_volume', 'ask_volume'
    ]
    last_record = start_record
    while True:
        num_records = (os.path.getsize(path) - HEADER_SIZE) // RECORD_SIZE
        if num_records > last_record:
            with open(path, 'rb') as f:
                f.seek(HEADER_SIZE + last_record * RECORD_SIZE)
                for _ in range(num_records - last_record):
                    rec_bytes = f.read(RECORD_SIZE)
                    if len(rec_bytes) < RECORD_SIZE:
                        break
                    dt, o, h, l, c, ntrades, tvol, bvol, avol = struct.unpack('<Q4f4I', rec_bytes)
                    try:
                        # Print raw timestamp value and try ns first
                        ts = pd.to_datetime(dt, unit='ns')
                        if ts.year < 2000 or ts.year > 2100:
                            # Try microseconds
                            ts = pd.to_datetime(dt, unit='us')
                            if ts.year < 2000 or ts.year > 2100:
                                # Try milliseconds
                                ts = pd.to_datetime(dt, unit='ms')
                        print(f"Raw timestamp: {dt}, Converted: {ts}")
                        yield {
                            'timestamp': ts,
                            'price': c,
                            'volume': tvol,
                            'num_trades': ntrades,
                            'bid_volume': bvol,
                            'ask_volume': avol
                        }
                    except Exception:
                        continue
            last_record = num_records
        time.sleep(poll_interval)
import struct
def load_sierra_chart_scid(path: str, start_record: int = 0, max_records: int = None) -> pd.DataFrame:
    """
    Load Sierra Chart .scid binary file as described in Reddit 'Hello World' guide.
    Returns DataFrame with columns: timestamp, open, high, low, close, num_trades, total_volume, bid_volume, ask_volume.
    Args:
        path: Path to .scid file
        start_record: Record index to start reading from (default 0)
        max_records: Max number of records to read (default: all)
    """
    HEADER_SIZE = 56
    RECORD_SIZE = 40
    columns = [
        'timestamp', 'open', 'high', 'low', 'close',
        'num_trades', 'total_volume', 'bid_volume', 'ask_volume'
    ]
    records = []
    skipped = 0
    with open(path, 'rb') as f:
        header = f.read(HEADER_SIZE)
        # Validate header
        id_bytes, header_size, record_size, version, unused1, utc_start, reserve = struct.unpack('<4s2I2HI36s', header)
        if id_bytes != b'SCID' or header_size != HEADER_SIZE or record_size != RECORD_SIZE:
            raise ValueError('Not a valid Sierra Chart .scid file')
        # Seek to start_record
        f.seek(HEADER_SIZE + start_record * RECORD_SIZE)
        num_records = (os.path.getsize(path) - HEADER_SIZE) // RECORD_SIZE
        if max_records is not None:
            num_to_read = min(max_records, num_records - start_record)
        else:
            num_to_read = num_records - start_record
        # Diagnostics: print first 20 raw timestamps
        raw_timestamps = []
        f.seek(HEADER_SIZE + start_record * RECORD_SIZE)
        for i in range(min(20, num_to_read)):
            rec_bytes = f.read(RECORD_SIZE)
            if len(rec_bytes) < RECORD_SIZE:
                break
            dt, *_ = struct.unpack('<Q4f4I', rec_bytes)
            raw_timestamps.append(dt)
        print(f"First {len(raw_timestamps)} raw timestamps: {raw_timestamps}")
        # Reset file pointer for main loop
        f.seek(HEADER_SIZE + start_record * RECORD_SIZE)
        for _ in range(num_to_read):
            rec_bytes = f.read(RECORD_SIZE)
            if len(rec_bytes) < RECORD_SIZE:
                break
            dt, o, h, l, c, ntrades, tvol, bvol, avol = struct.unpack('<Q4f4I', rec_bytes)
            # Try microsecond timestamp first, fallback to ms and ns
            ts = None
            try:
                ts = pd.to_datetime(dt, unit='us')
                if ts.year < 2000 or ts.year > 2100:
                    raise ValueError('Out of bounds us timestamp')
            except Exception:
                try:
                    ts = pd.to_datetime(dt, unit='ms')
                    if ts.year < 2000 or ts.year > 2100:
                        raise ValueError('Out of bounds ms timestamp')
                except Exception:
                    try:
                        ts = pd.to_datetime(dt, unit='ns')
                        if ts.year < 2000 or ts.year > 2100:
                            skipped += 1
                            continue
                    except Exception:
                        skipped += 1
                        continue
            records.append([ts, o, h, l, c, ntrades, tvol, bvol, avol])
    df = pd.DataFrame(records, columns=columns)
    if skipped > 0:
        print(f"Skipped {skipped} records with invalid timestamps.")
    return df
import os
import glob
def load_sierra_chart_trades(path: str) -> pd.DataFrame:
    """
    Load Sierra Chart trade data from a CSV or TXT file (or directory of files).
    Returns DataFrame with columns: timestamp, price, quantity, side.
    Autodetects common Sierra Chart layouts.
    """
    files = []
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.csv')) + glob.glob(os.path.join(path, '*.txt'))
    else:
        files = [path]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Try to autodetect columns
        if {'Date','Time','Price','Volume','Bid/Ask'}.issubset(df.columns):
            # Combine Date and Time to timestamp
            df['timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            df['price'] = df['Price']
            df['quantity'] = df['Volume']
            df['side'] = df['Bid/Ask'].map({'B':'buy','A':'sell','Bid':'buy','Ask':'sell'}).fillna('unknown')
            dfs.append(df[['timestamp','price','quantity','side']])
        else:
            # Fallback: try to use first 4 columns
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns and 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            else:
                df['timestamp'] = pd.to_datetime(df.iloc[:,0].astype(str) + ' ' + df.iloc[:,1].astype(str))
            df['price'] = df.get('price', df.iloc[:,2])
            df['quantity'] = df.get('volume', df.get('qty', df.iloc[:,3]))
            df['side'] = df.get('bid/ask', df.get('side', 'unknown'))
            dfs.append(df[['timestamp','price','quantity','side']])
    if dfs:
        result = pd.concat(dfs).sort_values('timestamp').reset_index(drop=True)
        return result
    else:
        return pd.DataFrame(columns=['timestamp','price','quantity','side'])

def load_sierra_chart_depth(path: str, levels: int = 5) -> pd.DataFrame:
    """
    Load Sierra Chart market depth (order book) from CSV/TXT file(s).
    Returns DataFrame with columns: timestamp, bid/ask prices and sizes for each level.
    """
    files = []
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.csv')) + glob.glob(os.path.join(path, '*.txt'))
    else:
        files = [path]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Try to autodetect columns for up to N levels
        # Example: BidPrice1, BidSize1, AskPrice1, AskSize1, ...
        cols = []
        for lvl in range(1, levels+1):
            cols += [f'BidPrice{lvl}', f'BidSize{lvl}', f'AskPrice{lvl}', f'AskSize{lvl}']
        base_cols = ['Date','Time']
        if set(base_cols + cols).issubset(df.columns):
            df['timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            out = df[['timestamp'] + cols]
            dfs.append(out)
        else:
            # Fallback: try to use first columns
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns and 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            else:
                df['timestamp'] = pd.to_datetime(df.iloc[:,0].astype(str) + ' ' + df.iloc[:,1].astype(str))
            # Try to extract bid/ask columns
            out_cols = ['timestamp']
            for lvl in range(1, levels+1):
                for prefix in ['bidprice','bidsize','askprice','asksize']:
                    col = f'{prefix}{lvl}'
                    if col in df.columns:
                        out_cols.append(col)
            out = df[out_cols]
            dfs.append(out)
    if dfs:
        result = pd.concat(dfs).sort_values('timestamp').reset_index(drop=True)
        return result
    else:
        return pd.DataFrame(['timestamp'] + cols)
import time
from datetime import datetime, timedelta
def download_binance_trades_range(symbol: str = "BTCUSDT", months: int = 6, out_csv: str = "btc_trades_6mo.csv") -> pd.DataFrame:
    """Download last N months of trades for a symbol from Binance US, paginating as needed.

    Parameters
    ----------
    symbol: str
        Trading pair symbol (e.g., 'BTCUSDT').
    months: int
        Number of months to go back.
    out_csv: str
        Path to save the resulting CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame of all trades in the range.
    """
    url = "https://api.binance.us/api/v3/aggTrades"
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=30*months)).timestamp() * 1000)
    all_trades = []
    fetch_start = start_time
    print(f"Downloading BTCUSDT trades from {months} months ago to now...")
    while fetch_start < end_time:
        params = {
            "symbol": symbol,
            "limit": 1000,
            "startTime": fetch_start,
            "endTime": end_time
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} {resp.text}")
            break
        batch = resp.json()
        if not batch:
            break
        df = pd.DataFrame(batch)
        df["timestamp"] = pd.to_datetime(df["T"], unit="ms")
        df = df.set_index("timestamp")
        # Only keep trades within the range
        df = df[(df.index >= pd.to_datetime(start_time, unit="ms")) & (df.index <= pd.to_datetime(end_time, unit="ms"))]
        if not df.empty:
            all_trades.append(df)
            # Move fetch_start forward to the last trade's timestamp + 1 ms
            fetch_start = int(df.index[-1].value // 1_000_000) + 1
        else:
            # If no trades, jump forward by 1 minute to avoid infinite loop
            fetch_start += 60_000
        # Sleep to avoid rate limits
        time.sleep(0.2)
    if all_trades:
        result = pd.concat(all_trades)
        result.to_csv(out_csv)
        print(f"Saved {len(result)} trades to {out_csv}")
        return result
    else:
        print("No trades found in the requested range.")
        return pd.DataFrame()
import requests
def download_binance_trades(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """Download recent trades for a symbol from Binance US public API (aggTrades endpoint).

    Parameters
    ----------
    symbol: str
        Trading pair symbol (e.g., 'BTCUSDT').
    limit: int
        Number of trades to fetch (max 1000 per request).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: price, qty, first tradeId, last tradeId, timestamp, isBuyerMaker, etc.
    """
    url = f"https://api.binance.us/api/v3/aggTrades"
    params = {"symbol": symbol, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        if "T" in df.columns:
            df["timestamp"] = pd.to_datetime(df["T"], unit="ms")
            df = df.set_index("timestamp")
        return df
    except requests.exceptions.HTTPError as e:
        print(f"Binance US API error: {e}. This may be due to geo-blocking or compliance restrictions.")
        return pd.DataFrame()
"""Data access helpers for the trading strategy tutorial project.

The Reddit post referenced by the user focuses on building a
machine-learning driven strategy on top of a liquid ETF.  To keep the
repository self-contained we only depend on `yfinance`, which is widely
available and easy to install.  The ``download_price_history`` function
wraps the yfinance API and returns a tidy ``pandas.DataFrame`` ready to be
fed into the rest of the pipeline.
"""

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
import pathlib
import glob
import os

try:  # pragma: no cover - yfinance is optional in some environments
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The yfinance package is required to download market data. "
        "Install it with `pip install yfinance`."
    ) from exc


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for the historical data download."""

    tickers: Iterable[str]
    start: _dt.date
    end: _dt.date | None = None
    interval: str = "1d"

    @property
    def tickers_list(self) -> List[str]:
        """Return the tickers as a list regardless of input type."""

        return list(self.tickers)


def download_price_history(config: DownloadConfig) -> pd.DataFrame:
    """Download adjusted OHLCV bars for the requested universe.

    Parameters
    ----------
    config:
        :class:`DownloadConfig` instance describing the assets, time span,
        and resolution of the data download.

    Returns
    -------
    pandas.DataFrame
        ``DataFrame`` indexed by ``DatetimeIndex`` containing the adjusted
        OHLCV data for all tickers in the configuration.  Columns follow the
        standard yfinance multi-index layout (``(field, ticker)``).
    """

    if config.end is None:
        end = _dt.date.today()
    else:
        end = config.end

    data = yf.download(
        tickers=config.tickers_list,
        start=config.start,
        end=end,
        interval=config.interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise ValueError(
            "No market data was returned. Check the ticker symbols and the date range."
        )

    # Ensure DatetimeIndex and sort just to be safe.
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    return data


def _read_and_concat_csvs(directory: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Read all CSVs matching pattern in ``directory`` and concat into one DataFrame.

    Tolerant to empty directories (returns empty DataFrame). Attempts to parse
    common timestamp columns and sets a DatetimeIndex when possible.
    """

    p = pathlib.Path(directory)
    if not p.exists() or not p.is_dir():
        return pd.DataFrame()

    files = sorted(glob.glob(str(p / pattern)))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        # Try to find a timestamp-like column
        for ts_col in ("timestamp", "time", "ts", "datetime"):
            if ts_col in df.columns:
                try:
                    df[ts_col] = pd.to_datetime(df[ts_col])
                    df = df.set_index(ts_col)
                except Exception:
                    pass
                break

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0)
    # If index is not a DatetimeIndex, attempt to coerce
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            # leave as-is
            pass

    out = out.sort_index()
    return out


def load_tick_data(directory: str) -> pd.DataFrame:
    """Load tick-level (trade-by-trade) data from a directory.

    The function expects CSV files in ``directory`` (for example, `/mnq` or
    `/nq`) where each CSV contains per-tick rows. Common column names are
    accommodated (``timestamp``, ``time``, ``price``, ``size``). The result
    is a DataFrame indexed by timestamp with columns left as-is.
    """

    df = _read_and_concat_csvs(directory)
    return df


def load_current_trades(directory: str) -> pd.DataFrame:
    """Load a snapshot of current trade state (one-row-per-instrument) from a directory.

    Useful for ingesting the current live trade/tick summary. The function
    looks for CSVs in ``directory`` and returns a concatenated DataFrame.
    """

    df = _read_and_concat_csvs(directory)
    return df


__all__ = ["DownloadConfig", "download_price_history"]
