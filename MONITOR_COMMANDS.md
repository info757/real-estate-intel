# Monitoring Commands

## Use `python3` (not `python`)

**From project directory: `/Users/williamholt/real-estate-intel`**

### Quick Status Check
```bash
cd /Users/williamholt/real-estate-intel
python3 scripts/monitor_training.py
```

### Live Monitor (auto-refresh)
```bash
cd /Users/williamholt/real-estate-intel
python3 scripts/monitor_training_live.py
```

### Count DOM Metrics
```bash
cd /Users/williamholt/real-estate-intel
python3 scripts/count_dom_metrics.py
```

### Watch Log File
```bash
cd /Users/williamholt/real-estate-intel
tail -f training.log
```

### Check if Training is Running
```bash
cd /Users/williamholt/real-estate-intel
ps -p $(cat training.pid 2>/dev/null) && echo "✅ Running" || echo "❌ Not running"
```

### If Using Virtual Environment
```bash
cd /Users/williamholt/real-estate-intel
source venv/bin/activate
python scripts/monitor_training.py
```
