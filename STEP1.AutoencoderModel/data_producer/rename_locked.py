import os
import sys

def recover_locked_files(cache_dir, dry_run=True):
    locked = [
        f for f in os.listdir(cache_dir)
        if f.startswith('_locked_') and f.endswith('.pt')
    ]
    
    if not locked:
        print("No locked files found.")
        return
    
    print(f"Found {len(locked)} locked files.")
    total_mb = 0
    for f in locked:
        locked_path = os.path.join(cache_dir, f)
        recovered_name = f[len('_locked_'):]
        recovered_path = os.path.join(cache_dir, recovered_name)
        size_mb = os.path.getsize(locked_path) / (1024 ** 2)
        total_mb += size_mb
        
        if dry_run:
            print(f"  [DRY RUN] {f} -> {recovered_name} ({size_mb:.1f} MB)")
        else:
            os.rename(locked_path, recovered_path)
            print(f"  Recovered: {f} -> {recovered_name} ({size_mb:.1f} MB)")
    
    print(f"\nTotal {'recoverable' if dry_run else 'recovered'}: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")

if __name__ == "__main__":
    cache_dir = "/scratch/rpinise1/MultiTumorSynthesis/AutoencoderCache"
    
    dry_run = "--confirm" not in sys.argv
    if dry_run:
        print("Running in dry-run mode. Pass --confirm to actually rename.\n")
    
    recover_locked_files(cache_dir, dry_run=dry_run)