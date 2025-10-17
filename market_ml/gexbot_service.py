import os
import json
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from market_ml.gexbot import stream_gex_realtime, load_historical_gex, probe_gexbot_key
from market_ml.config import Config

config = Config()
API_KEY = config.GEXBOT_API_KEY
SYMBOLS = ['VIX', 'ES_SPX', 'NQ_NDX', 'SPY', 'QQQ', 'AAPL']
AGGREGATION = config.AGGREGATION
MAX_REQUESTS_PER_SEC = config.MAX_REQUESTS_PER_SEC

class GexbotService:
    def save_historical(self, out_dir=None):
        if out_dir is None:
            out_dir = config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        for symbol, dfs in self.historical_data.items():
            if dfs:
                df_all = pd.concat(dfs, ignore_index=False)
                # Ensure we have all the important columns
                output_path = os.path.join(out_dir, f"{symbol}_gex_snapshot.csv")
                df_all.to_csv(output_path)
                print(f"Saved GEX snapshot for {symbol} to {output_path}")
                # Also log key metrics
                if not df_all.empty:
                    latest = df_all.iloc[-1]
                    metrics = {
                        'price': latest.get('price'),
                        'zero_gamma': latest.get('zero_gamma'),
                        'sum_gex_oi': latest.get('sum_gex_oi'),
                        'major_call_oi': latest.get('major_call_oi'),
                        'major_put_oi': latest.get('major_put_oi'),
                    }
                    print(f"  Latest {symbol}: {metrics}")
    
    def save_summary(self, out_dir=None):
        """Save a compact summary of latest levels for all symbols."""
        if out_dir is None:
            out_dir = config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        
        summary_rows = []
        for symbol, dfs in self.historical_data.items():
            if dfs:
                df_all = pd.concat(dfs, ignore_index=False)
                if not df_all.empty:
                    latest = df_all.iloc[-1]
                    summary_rows.append({
                        'symbol': symbol,
                        'timestamp': df_all.index[-1] if hasattr(df_all.index, '__getitem__') else None,
                        'price': latest.get('price'),
                        'zero_gamma': latest.get('zero_gamma'),
                        'sum_gex_oi': latest.get('sum_gex_oi'),
                        'sum_gex_vol': latest.get('sum_gex_vol'),
                        'major_call_oi': latest.get('major_call_oi'),
                        'major_call_vol': latest.get('major_call_vol'),
                        'major_put_oi': latest.get('major_put_oi'),
                        'major_put_vol': latest.get('major_put_vol'),
                    })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(out_dir, 'gex_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSaved GEX summary to {summary_path}")
            return summary_df
        return None

    def fetch_and_append_snapshot(self):
        """Fetch current snapshot and append to historical data for building time-series."""
        for symbol in self.symbols:
            try:
                # Fetch current zero levels (use wide date range to avoid filtering out data)
                df = load_historical_gex(
                    symbol=symbol,
                    start_date='2000-01-01',
                    end_date='2099-12-31',
                    aggregation='zero',
                    api_key=self.api_key
                )
                if not df.empty:
                    # Append to existing data
                    if symbol in self.historical_data and self.historical_data[symbol]:
                        self.historical_data[symbol].append(df)
                    else:
                        self.historical_data[symbol] = [df]
            except Exception as e:
                print(f"Error fetching snapshot for {symbol}: {e}")

    def save_realtime(self, out_dir=None):
        if out_dir is None:
            out_dir = config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        for symbol, data in self.realtime_data.items():
            with open(os.path.join(out_dir, f"{symbol}_gex_realtime.json"), 'w') as f:
                json.dump(data, f, default=str)
            print(f"Saved realtime data for {symbol} to {out_dir}/{symbol}_gex_realtime.json")
    def __init__(self, symbols, api_key, aggregation='1d', max_rps=20):
        self.symbols = symbols
        self.api_key = api_key
        self.aggregation = aggregation
        self.max_rps = max_rps
        self.historical_data = {}
        self.realtime_data = {}
        self._stop_event = threading.Event()

    def fetch_historical(self, months=6):
        # Use timezone-aware now (UTC)
        try:
            from datetime import UTC
            end_date = datetime.now(UTC)
        except Exception:
            # Fallback if UTC not available (older Python)
            end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30*months)
        
        # For now, use classic/zero endpoint (single snapshot per symbol)
        # classic/1d may require higher tier or different auth
        for symbol in self.symbols:
            try:
                df = load_historical_gex(
                    symbol=symbol,
                    start_date=start_date.date().isoformat(),
                    end_date=end_date.date().isoformat(),
                    aggregation='zero',  # Use zero endpoint for current levels
                    api_key=self.api_key
                )
                self.historical_data[symbol] = [df]
                if not df.empty:
                    print(f"Fetched GEX snapshot for {symbol}: {len(df)} row(s), cols: {list(df.columns)}")
                else:
                    print(f"No data returned for {symbol}")
            except Exception as e:
                print(f"Error fetching {symbol} snapshot: {e}")
        print("Historical GEX data fetch complete.")

    def start_realtime(self):
        def handle_gex_update(data):
            symbol = data.get('symbol')
            if symbol:
                self.realtime_data[symbol] = data
            print(f"[Realtime] {symbol}: {data}")
        threading.Thread(
            target=stream_gex_realtime,
            args=(self.symbols,),
            kwargs={'callback': handle_gex_update, 'api_key': self.api_key},
            daemon=True
        ).start()
        print("Started GEXBOT realtime streaming.")

    def stop(self):
        self._stop_event.set()

if __name__ == '__main__':
    # Validate API key before making heavy calls
    probe = probe_gexbot_key(API_KEY)
    if not probe.get("valid"):
        print(f"GEXBOT API key check failed: {probe.get('detail')}")
        raise SystemExit(1)
    else:
        print(f"GEXBOT API key check passed: {probe.get('detail')}")

    service = GexbotService(SYMBOLS, API_KEY, AGGREGATION, MAX_REQUESTS_PER_SEC)
    
    # Fetch initial snapshot
    print("Fetching initial GEX snapshot...")
    service.fetch_historical(months=2)
    service.save_historical()
    service.save_summary()
    
    print("\nStarting realtime GEX streaming...")
    service.start_realtime()
    
    # Periodic polling to build historical time-series
    # Poll every 5 minutes to build up 2 months of data over time
    POLL_INTERVAL_SECONDS = 300  # 5 minutes
    print(f"\nService running. Polling every {POLL_INTERVAL_SECONDS}s to build historical time-series.")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching new snapshot...")
            service.fetch_and_append_snapshot()
            service.save_historical()
            service.save_summary()
            service.save_realtime()
    except KeyboardInterrupt:
        print("\nStopping service...")
        service.stop()
