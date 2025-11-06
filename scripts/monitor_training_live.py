"""Live monitoring of training progress with auto-refresh."""
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def monitor_live():
    """Monitor training with live updates."""
    cache_dir = Path("cache/listings")
    log_file = Path("training.log")
    pid_file = Path("training.pid")
    
    print("="*80)
    print("LIVE TRAINING MONITOR")
    print("="*80)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            # Clear screen (works on most terminals)
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("="*80)
            print(f"LIVE TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
            print("="*80)
            print()
            
            # Check if training is running
            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
                import subprocess
                try:
                    result = subprocess.run(['ps', '-p', str(pid)], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ Training is RUNNING (PID: {pid})")
                    else:
                        print("❌ Training process not running")
                except:
                    print("⚠️  Cannot check process status")
            else:
                print("⚠️  No PID file found")
            
            print()
            
            # Check cache
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("listings_*.json"))
                total_listings = 0
                zip_stats = []
                
                for cache_file in cache_files:
                    try:
                        with open(cache_file, 'r') as f:
                            listings = json.load(f)
                        zip_code = cache_file.stem.split('_')[1]
                        count = len(listings)
                        total_listings += count
                        with_dom = sum(1 for l in listings if l.get('dom_to_pending') is not None)
                        zip_stats.append({'zip': zip_code, 'count': count, 'with_dom': with_dom})
                    except:
                        pass
                
                print(f"📊 Cache Status: {len(cache_files)} ZIPs processed")
                print(f"   Total listings: {total_listings}")
                if zip_stats:
                    print(f"   With dom_to_pending: {sum(s['with_dom'] for s in zip_stats)}")
                    print()
                    print("ZIP Progress:")
                    for stat in sorted(zip_stats, key=lambda x: x['zip']):
                        print(f"   {stat['zip']}: {stat['count']} listings ({stat['with_dom']} with DOM)")
            else:
                print("📊 Cache: No data yet")
            
            print()
            
            # Show recent log entries
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                    print("Recent Log Entries:")
                    print("-" * 80)
                    for line in lines[-10:]:
                        print(line.rstrip())
                except:
                    pass
            
            print()
            print("="*80)
            print("Refreshing in 10 seconds... (Ctrl+C to stop)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == '__main__':
    try:
        monitor_live()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
