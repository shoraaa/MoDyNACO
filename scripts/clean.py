#!/usr/bin/env python3
"""
Clean old experiments safely.

DRY RUN BY DEFAULT! Use --force to actually delete.

Usage:
    python scripts/clean.py --older-than 30d --dry-run
    python scripts/clean.py --older-than 90d --force
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import shutil


def parse_age_spec(spec: str) -> timedelta:
    """Parse age specification like '30d' or '7d'."""
    if spec.endswith('d'):
        days = int(spec[:-1])
        return timedelta(days=days)
    elif spec.endswith('h'):
        hours = int(spec[:-1])
        return timedelta(hours=hours)
    else:
        # Assume days
        return timedelta(days=int(spec))


def find_old_experiments(older_than: timedelta):
    """Find experiments older than specified age."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return []
    
    cutoff = datetime.now() - older_than
    old_exps = []
    
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check modification time of logs directory or config
        mtime = None
        log_dir = exp_dir / "logs"
        if log_dir.exists():
            mtime = datetime.fromtimestamp(log_dir.stat().st_mtime)
        else:
            mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)
        
        if mtime < cutoff:
            old_exps.append((exp_dir, mtime))
    
    return sorted(old_exps, key=lambda x: x[1])


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except:
                pass
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--older-than", type=str, default="30d",
                       help="Delete experiments older than this (e.g., '30d', '7d')")
    parser.add_argument("--free-space", type=str, default=None,
                       help="Delete oldest experiments until this much free space (e.g., '10GB')")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be deleted (default)")
    parser.add_argument("--force", action="store_true",
                       help="Actually delete (overrides --dry-run)")
    args = parser.parse_args()
    
    # Determine actual dry-run setting
    dry_run = not args.force
    
    # Parse age spec
    age_threshold = parse_age_spec(args.older_than)
    print(f"Looking for experiments older than {args.older_than} ({age_threshold.days} days)...")
    
    old_exps = find_old_experiments(age_threshold)
    
    if not old_exps:
        print("No old experiments found.")
        return
    
    print(f"Found {len(old_exps)} old experiments:")
    total_size = 0
    for exp_dir, mtime in old_exps:
        size = get_dir_size(exp_dir)
        total_size += size
        print(f"  {exp_dir.name} (modified {mtime.strftime('%Y-%m-%d')}, {format_size(size)})")
    
    print(f"\nTotal space that would be freed: {format_size(total_size)}")
    
    if dry_run:
        print("\nDRY RUN - no files deleted.")
        print("To actually delete, run with --force")
    else:
        print("\nDELETING (--force specified)...")
        for exp_dir, mtime in old_exps:
            try:
                shutil.rmtree(exp_dir)
                print(f"  Deleted {exp_dir}")
            except Exception as e:
                print(f"  Error deleting {exp_dir}: {e}")
        print(f"\nDeleted {len(old_exps)} experiments, freed {format_size(total_size)}")


if __name__ == "__main__":
    main()
