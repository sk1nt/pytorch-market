# GEX + Order Flow + MACD Strategy - Complete Setup

## ✅ What's Ready

### Strategy Components
1. **MACD Indicator** - Default settings (12, 26, 9) on 1-minute bars
2. **Order Flow Analysis** - Cumulative delta from SCID tick data
3. **GEX Integration** - Zero gamma levels and major strikes
4. **Complete Backtesting Framework** - With transaction costs and performance metrics

### Data Available
- **SCID Files**: 1.6M+ MNQ records, 1.1M+ NQ records (Mar-Oct 2025)
- **GEX Data**: Backfill historical data (limited market hours data so far)
- **Trading Utilities**: Position sizing, PnL, equity curves, risk metrics

### Files Created
```
market_ml/
├── gex_orderflow_strategy.py    # Main strategy implementation
├── load_trades.py                # Trade loading utilities
├── utils.py                      # Trading utilities (existing)
└── data.py                       # SCID parser (existing)

examples/
└── run_gex_orderflow_backtest.py # Example runner with plots

docs/
└── gex_orderflow_strategy.md    # Complete documentation
```

## 🚀 How to Run

### Option 1: Quick Start (Recommended)
```bash
python examples/run_gex_orderflow_backtest.py
```

This automatically:
- Loads MNQZ25 SCID data
- Loads available GEX data (backfill or intraday)
- Runs complete backtest
- Generates CSV results + PNG visualization

### Option 2: Command Line
```bash
python -m market_ml.gex_orderflow_strategy \
  --scid /mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid \
  --gex outputs/backfill/NQ_NDX_gex_historical.csv \
  --output outputs/backtest_results.csv \
  --capital 10000 \
  --fee 0.0004
```

### Option 3: Python Script
```python
from market_ml.gex_orderflow_strategy import run_full_backtest

results, metrics = run_full_backtest(
    scid_file='/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid',
    gex_file='outputs/backfill/NQ_NDX_gex_historical.csv',
    output_file='outputs/my_backtest.csv'
)

print(f"Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
```

## 📊 Strategy Logic

### Entry Signals

**LONG** when ALL conditions met:
- MACD line crosses above signal line ✓
- Price is above zero gamma level ✓
- Cumulative delta is positive ✓

**SHORT** when ALL conditions met:
- MACD line crosses below signal line ✓
- Price is below zero gamma level ✓
- Cumulative delta is negative ✓

### Exit
- Opposite signal (long → short or vice versa)
- Position held until next signal

## 📈 Output

### CSV Results (`outputs/gex_orderflow_backtest.csv`)
Contains every 1-minute bar with:
- Timestamp, price (OHLC)
- MACD, MACD signal, MACD histogram
- Delta, cumulative delta
- Zero gamma level
- Position, equity, PnL
- All GEX features

### Visualization (`outputs/gex_orderflow_backtest.png`)
4-panel chart showing:
1. **Price & Signals** - Entry/exit markers + zero gamma overlay
2. **MACD** - Indicator with crossover points
3. **Cumulative Delta** - Order flow pressure
4. **Equity Curve** - Strategy performance

### Performance Metrics
```
Period: 2025-03-13 to 2025-10-17
Initial Capital: $10,000.00
Final Equity: $XX,XXX.XX
Total Return: +X.XX%
Number of Trades: XXX
Win Rate: XX.X%
Max Drawdown: -X.XX%
Sharpe Ratio: X.XX
```

## 🔧 Customization

### Change MACD Parameters
```python
# In gex_orderflow_strategy.py, modify calculate_macd():
df = calculate_macd(df, fast=8, slow=21, signal=5)  # Faster settings
```

### Adjust Delta Threshold
```python
# In generate_signals():
long_delta = df['cum_delta'] > 2000   # Require stronger buying
short_delta = df['cum_delta'] < -2000  # Require stronger selling
```

### Add Stop Loss / Take Profit
```python
from market_ml.gex_orderflow_strategy import apply_risk_management

results = apply_risk_management(
    results,
    stop_loss_pct=0.015,     # 1.5% stop
    take_profit_pct=0.025    # 2.5% target
)
```

### Different Instrument
```python
# NQ instead of MNQ
results, metrics = run_full_backtest(
    scid_file='/mnt/c/SierraChart/Data/NQZ25_FUT_CME.scid',
    gex_file='outputs/backfill/NQ_NDX_gex_historical.csv'
)

# ES (S&P 500)
results, metrics = run_full_backtest(
    scid_file='/mnt/c/SierraChart/Data/ESZ25_FUT_CME.scid',
    gex_file='outputs/backfill/ES_SPX_gex_historical.csv'
)
```

## 📚 Key Functions Reference

### Strategy Pipeline
- `calculate_macd(df)` - Add MACD indicator
- `calculate_order_flow_features(df)` - Add delta metrics
- `add_gex_features(scid_df, gex_df)` - Merge GEX levels
- `resample_to_1min(df)` - Convert ticks to 1-min bars
- `generate_signals(df)` - Create entry/exit signals
- `backtest_strategy(df)` - Run backtest with costs
- `run_full_backtest()` - Complete end-to-end pipeline

### Utility Functions (from `utils.py`)
- `add_compounding_trade_sizing()` - Position sizing
- `add_trade_gross_pnl()` - Raw PnL
- `add_transaction_fees()` - Trading costs
- `add_trade_net_pnl()` - Net PnL after fees
- `add_equity_curves()` - Account equity tracking
- `win_rate()` - Win percentage

### Data Loading (from `load_trades.py`)
- `load_trades_from_csv()` - Load existing trades
- `load_trades_from_broker_csv()` - Import broker data
- `load_trades_from_scid()` - Generate from SCID strategy
- `combine_trades_with_gex()` - Merge trades + GEX

## ⚠️ Important Notes

### GEX Data Availability
- **Backfill data**: Currently available, updated hourly
- **Intraday data**: Limited (market hours collector just started)
- **Best results**: After several days of collection

Check status:
```bash
./check_status.sh
```

### SCID File Size
- Files are large (1M+ records)
- Full backtest may take several minutes
- Consider testing on subset first:
```python
from market_ml.data import load_sierra_chart_scid
df = load_sierra_chart_scid(scid_path, max_records=100000)
```

### Market Hours
- Strategy works best during liquid hours
- Consider filtering to RTH (9:30 AM - 4:00 PM ET)
- Overnight moves may affect results

## 🎯 Next Steps

1. **Run first backtest** to see baseline performance
2. **Review signals** in the output CSV
3. **Adjust parameters** based on results
4. **Test on different instruments** (MNQ, NQ, ES, SPY)
5. **Add risk management** (stops, targets)
6. **Forward test** once satisfied with backtest
7. **Paper trade** before going live

## 📖 Documentation

- `docs/gex_orderflow_strategy.md` - Full strategy guide
- `docs/market_hours_collection.md` - GEX data collection
- `docs/scid_timestamp_fix.md` - SCID file format
- `.github/copilot-instructions.md` - Project architecture

## 🔍 Troubleshooting

**"No GEX data available"**
→ Use backfill data or wait for market hours collector

**"SCID file not found"**
→ Check path: `ls /mnt/c/SierraChart/Data/*.scid`

**"Memory error"**
→ Use max_records parameter to limit data size

**"Division by zero"**
→ Ensure GEX data has valid timestamps overlapping SCID data

---

**Ready to backtest!** Run: `python examples/run_gex_orderflow_backtest.py`
