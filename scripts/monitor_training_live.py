"""Live monitoring of training progress with auto-refresh."""
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _read_training_log(log_file: Path, lines: int = 15) -> list[str]:
    try:
        with open(log_file, 'r') as f:
            return f.readlines()[-lines:]
    except Exception:
        return []


def _collect_cache_stats(cache_dir: Path) -> dict:
    stats = {}
    if not cache_dir.exists():
        return stats

    for entry in cache_dir.iterdir():
        if entry.is_dir():
            source = entry.name
            files = list(entry.glob('listings_*.json'))
        elif entry.is_file() and entry.name.startswith('listings_'):
            source = 'legacy'
            files = [entry]
        else:
            continue

        for cache_file in files:
            try:
                with open(cache_file, 'r') as f:
                    listings = json.load(f)
            except Exception:
                continue

            stem_parts = cache_file.stem.split('_')
            zip_code = stem_parts[1] if len(stem_parts) > 1 else 'unknown'
            key = (source, zip_code)
            stats.setdefault(key, {'count': 0, 'with_dom': 0})
            stats[key]['count'] += len(listings)
            stats[key]['with_dom'] += sum(1 for l in listings if l.get('dom_to_pending') is not None)

    return stats


def monitor_live(refresh_interval: int = 10):
    cache_dir = Path('cache/listings')
    log_file = Path('training.log')
    pid_file = Path('training.pid')

    print("=" * 80)
    print("LIVE TRAINING MONITOR")
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring")
    print()

    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')

            print("=" * 80)
            print(f"LIVE TRAINING MONITOR - {datetime.now():%H:%M:%S}")
            print("=" * 80)
            print()

            # Process status
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                except ValueError:
                    pid = None
                if pid:
                    try:
                        result = subprocess.run(['ps', '-p', str(pid)], capture_output=True, text=True)
                    except Exception:
                        result = None
                    if result and result.returncode == 0:
                        print(f"✅ Training is RUNNING (PID: {pid})")
                    else:
                        print(f"❌ Process {pid} not running")
                else:
                    print("⚠️  PID file unreadable")
            else:
                print("⚠️  No PID file found")

            print()

            # Cache summary
            stats = _collect_cache_stats(cache_dir)
            if stats:
                total_zips = len(stats)
                total_listings = sum(item['count'] for item in stats.values())
                total_dom = sum(item['with_dom'] for item in stats.values())
                print(f"📊 Cache Status: {total_zips} ZIP snapshots across {len({src for src, _ in stats})} sources")
                print(f"   Total listings: {total_listings}")
                print(f"   Listings with dom_to_pending: {total_dom}")
                print()
                print("ZIP Progress:")
                for (source, zip_code), item in sorted(stats.items(), key=lambda s: (s[0][0], s[0][1])):
                    print(f"   {zip_code} [{source}]: {item['count']} listings ({item['with_dom']} with DOM)")
            else:
                print("📊 Cache: No data yet")

            print()

            # Recent log entries
            lines = _read_training_log(log_file)
            if lines:
                print("Recent Log Entries:")
                print("-" * 80)
                for line in lines:
                    print(line.rstrip())
            else:
                print("No training log entries yet")

            print()
            print("=" * 80)
            print(f"Refreshing in {refresh_interval} seconds... (Ctrl+C to stop)")
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == '__main__':
    import subprocess

    try:
        monitor_live()
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
