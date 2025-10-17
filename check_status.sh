#!/bin/bash
# Quick status check for all GEX data collection processes

echo "═══════════════════════════════════════════════════════════════"
echo "  GEX Data Collection Status"
echo "═══════════════════════════════════════════════════════════════"
echo

# Current time in EST
echo "📅 Current Time (EST):"
TZ='America/New_York' date "+%Y-%m-%d %H:%M:%S %Z (%A)"
echo

# Check running processes
echo "🔄 Running Processes:"
echo "-------------------------------------------------------------------"
ps aux | grep -E "backfill_gex|backfill_market_hours|gexbot_service" | grep -v grep | \
    awk '{printf "  PID %-7s %5s%% CPU  %s\n", $2, $3, substr($0, index($0,$11))}'
    
if [ $? -ne 0 ]; then
    echo "  ⚠️  No collection processes running"
fi
echo

# Check log files
echo "📊 Recent Activity:"
echo "-------------------------------------------------------------------"
for log in backfill.log intraday_zero.log intraday_full.log gexbot_service.log; do
    if [ -f "$log" ]; then
        echo "  📄 $log (last 3 lines):"
        tail -3 "$log" 2>/dev/null | sed 's/^/    /'
        echo
    fi
done

# Check collected data
echo "💾 Collected Data:"
echo "-------------------------------------------------------------------"

if [ -d "outputs/backfill" ]; then
    echo "  Hourly 24/7 collection (outputs/backfill/):"
    for file in outputs/backfill/*_gex_historical.csv; do
        if [ -f "$file" ]; then
            lines=$(($(wc -l < "$file") - 1))  # Subtract header
            size=$(ls -lh "$file" | awk '{print $5}')
            name=$(basename "$file")
            printf "    %-35s %6s rows  %7s\n" "$name" "$lines" "$size"
        fi
    done
    echo
fi

if [ -d "outputs/intraday" ]; then
    echo "  Market hours collection (outputs/intraday/):"
    for file in outputs/intraday/*_intraday.csv; do
        if [ -f "$file" ]; then
            lines=$(($(wc -l < "$file") - 1))  # Subtract header
            size=$(ls -lh "$file" | awk '{print $5}')
            name=$(basename "$file")
            printf "    %-35s %6s rows  %7s\n" "$name" "$lines" "$size"
        fi
    done
    echo
fi

if [ -f "outputs/gex_summary.csv" ]; then
    echo "  Latest snapshot (outputs/gex_summary.csv):"
    tail -1 outputs/gex_summary.csv | awk -F',' '{print "    " $0}' | sed 's/,/  |  /g'
    echo
fi

# Market hours info
echo "📈 Market Hours:"
echo "-------------------------------------------------------------------"
echo "  Trading: Monday-Friday, 8:30 AM - 6:00 PM EST"
echo "  Minute-by-minute collection during market hours only"
echo

# Quick commands
echo "🔧 Quick Commands:"
echo "-------------------------------------------------------------------"
echo "  View live logs:       tail -f intraday_zero.log"
echo "  Stop all collectors:  pkill -f backfill"
echo "  Check this again:     bash check_status.sh"
echo

echo "═══════════════════════════════════════════════════════════════"
