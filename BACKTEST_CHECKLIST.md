# GEX + Order Flow + MACD Backtest Checklist

## ✅ Pre-Flight Check

### Data Ready?
- [x] SCID files loaded and parsed correctly
  - MNQZ25: 1,649,350 records ✓
  - NQZ25: 1,153,339 records ✓
- [x] GEX data available
  - Backfill data: outputs/backfill/*.csv ✓
  - Intraday data: Collecting (limited so far)
- [x] All utilities tested ✓

### Files Ready?
- [x] market_ml/gex_orderflow_strategy.py ✓
- [x] examples/run_gex_orderflow_backtest.py ✓
- [x] market_ml/load_trades.py ✓
- [x] Documentation complete ✓

## 🚀 Run Backtest

### Step 1: Simple Test Run
```bash
# Quick test to verify everything works
python examples/run_gex_orderflow_backtest.py
```

**Expected**: Script runs, generates results, creates plots
**Time**: 2-5 minutes (depending on data size)

### Step 2: Review Results
```bash
# Check output files
ls -lh outputs/gex_orderflow_backtest.*

# View performance summary
tail -20 outputs/gex_orderflow_backtest.csv

# Open visualization
# (Use image viewer or transfer PNG to view)
```

### Step 3: Analyze Performance
Check these metrics:
- [ ] Total Return %
- [ ] Win Rate
- [ ] Max Drawdown
- [ ] Sharpe Ratio
- [ ] Number of Trades

### Step 4: Review Signals
```bash
# Look at actual trade signals
python -c "
import pandas as pd
df = pd.read_csv('outputs/gex_orderflow_backtest.csv')
trades = df[df['signal'].diff().fillna(0) != 0]
print(trades[['timestamp', 'price', 'signal', 'macd', 'cum_delta', 'zero_gamma']].head(20))
"
```

## 🔧 Next Steps Based on Results

### If Performance is Good (>50% win rate, positive Sharpe)
- [ ] Test on different time periods
- [ ] Try other instruments (NQ, ES, SPY)
- [ ] Add risk management (stops, targets)
- [ ] Optimize parameters

### If Performance Needs Work
- [ ] Adjust MACD periods (try faster/slower)
- [ ] Change delta threshold (require stronger signal)
- [ ] Filter by volume (avoid low-liquidity periods)
- [ ] Add time-of-day filters (RTH only)

### If Strategy Shows Promise
- [ ] Walk-forward optimization
- [ ] Out-of-sample testing
- [ ] Paper trading setup
- [ ] Risk management rules

## 📊 Parameter Tuning Ideas

### MACD Settings
```python
# Faster (more signals)
calculate_macd(df, fast=8, slow=17, signal=9)

# Slower (fewer signals)
calculate_macd(df, fast=15, slow=30, signal=10)
```

### Delta Threshold
```python
# More aggressive (current)
long_delta = df['cum_delta'] > 0

# More conservative
long_delta = df['cum_delta'] > 1000

# Very conservative
long_delta = df['cum_delta'] > 2000
```

### GEX Distance
```python
# Current: Just above/below zero gamma
long_gex = df['above_zero_gamma']

# With buffer zone
long_gex = (df['close'] - df['zero_gamma']) > 5  # 5 points above

# Percentage based
long_gex = df['pct_from_zero_gamma'] > 0.1  # 0.1% above
```

## 💡 Customization Examples

### Add Volume Filter
```python
# In generate_signals():
long_volume = df['volume'] > df['volume'].rolling(20).mean()
df.loc[long_macd & long_gex & long_delta & long_volume, 'signal'] = 1
```

### Time-of-Day Filter
```python
# Only trade 9:30 AM - 3:30 PM ET
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['minute'] = pd.to_datetime(df['timestamp']).dt.minute

rth_hours = (
    ((df['hour'] == 9) & (df['minute'] >= 30)) |
    ((df['hour'] >= 10) & (df['hour'] <= 14)) |
    ((df['hour'] == 15) & (df['minute'] <= 30))
)

# Only generate signals during RTH
df.loc[~rth_hours, 'signal'] = 0
```

### Add Stop Loss
```python
results = apply_risk_management(
    results,
    stop_loss_pct=0.02,      # 2% stop
    take_profit_pct=0.03     # 3% target
)
```

## 📝 Results Log Template

```
Date: _____________
Instrument: _____________
Period: _____________ to _____________

Parameters:
- MACD: (12, 26, 9) or custom: _____________
- Delta Threshold: _____________
- Initial Capital: _____________
- Transaction Fee: _____________

Results:
- Total Return: _____________ %
- Number of Trades: _____________
- Win Rate: _____________ %
- Max Drawdown: _____________ %
- Sharpe Ratio: _____________

Notes:
_____________________________________
_____________________________________
_____________________________________

Next Test:
_____________________________________
```

## 🎯 Success Criteria

Before going live, verify:
- [ ] Positive returns over multiple test periods
- [ ] Win rate > 45%
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown < 15%
- [ ] Strategy logic makes sense
- [ ] Results are consistent across instruments
- [ ] Paper trading shows similar results

## 🔍 Common Issues

**"No trades generated"**
→ Conditions too strict, loosen delta threshold or MACD settings

**"Too many trades"**
→ Add filters (volume, time-of-day) or stricter conditions

**"Large drawdowns"**
→ Add stop losses, reduce position size, or tighten filters

**"Inconsistent results"**
→ Test on longer periods, check for data quality issues

---

Ready to run! Start with: `python examples/run_gex_orderflow_backtest.py`
