#!/usr/bin/env python3
"""
Migrate old checkpoints to new experiments structure.

DRY RUN BY DEFAULT! Use --execute to actually move files.

Usage:
    python scripts/migrate_checkpoints.py --dry-run
    python scripts/migrate_checkpoints.py --execute
"""

import argparse
from pathlib import Path
import shutil
import re
from datetime import datetime


def parse_checkpoint_filename(filename: str) -> dict:
    """Parse checkpoint filename to extract hyperparameters.
    
    Expected patterns:
    - tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt
    - cvrp_n1000_k32_ants100_H10_miniH100_rho0.5_mne12_ppo_lr5e-06_best.pt
    - bpp_n50_k32_ants20_H5_miniH5_rho0.9_lr0.0003_best.pt
    """
    params = {}
    name = filename.replace('.pt', '')
    
    # Extract problem type
    if name.startswith('tsp_'):
        params['problem'] = 'tsp'
    elif name.startswith('cvrp_'):
        params['problem'] = 'cvrp'
    elif name.startswith('bpp_'):
        params['problem'] = 'bpp'
    elif name.startswith('mkp_'):
        params['problem'] = 'mkp'
    elif name.startswith('op_'):
        params['problem'] = 'op'
    else:
        params['problem'] = 'unknown'
    
    # Extract n_node
    n_match = re.search(r'n(\d+)', name)
    if n_match:
        params['n_node'] = int(n_match.group(1))
    
    # Extract k_sparse
    k_match = re.search(r'k(\d+)', name)
    if k_match:
        params['k_sparse'] = int(k_match.group(1))
    
    # Extract n_ants
    ants_match = re.search(r'ants(\d+)', name)
    if ants_match:
        params['n_ants'] = int(ants_match.group(1))
    
    # Extract H and mini_H
    H_match = re.search(r'H(\d+)', name)
    if H_match:
        params['H'] = int(H_match.group(1))
    
    miniH_match = re.search(r'miniH(\d+)', name)
    if miniH_match:
        params['mini_H'] = int(miniH_match.group(1))
    
    # Extract rho
    rho_match = re.search(r'rho([\d.]+)', name)
    if rho_match:
        params['rho'] = float(rho_match.group(1))
    
    # Extract learning rate
    lr_match = re.search(r'lr([\d.e-]+)', name)
    if lr_match:
        lr_str = lr_match.group(1)
        params['lr'] = float(lr_str)
    
    # Determine if best checkpoint
    params['is_best'] = '_best' in filename
    
    return params


def generate_config_from_params(params: dict) -> dict:
    """Generate a minimal config dict from parsed checkpoint params."""
    # Default values for missing parameters
    defaults = {
        'algo': 'ppo',
        'epochs': 10,
        'device': 'cuda:0',
        'seed': 1234,
        'min_new_edges': 12,
        'no_local_search': False,
        'no_smooth_mmas': False,
        'no_extend_ls': False,
        'no_normalized_heuristic': False,
        'disable_heuristic': False,
        'no_logit_net': False,
        'no_dynamic_feats': False,
        'ablation_pheromone_features': False,
        'ablation_incumbent_features': False,
    }
    
    config = defaults.copy()
    config.update(params)
    
    # Remove internal metadata
    config.pop('is_best', None)
    
    return config


def migrate_checkpoints(dry_run: bool = True):
    """Migrate checkpoints from old structure to new experiments/."""
    old_dirs = [
        Path("checkpoints"),
        Path("checkpoints_extended"),
        Path("checkpoints_dynamic"),
        Path("checkpoints_static"),
        Path("checkpoints_compare"),
        Path("pretrained"),
    ]
    
    migrated = []
    skipped = []
    
    for old_dir in old_dirs:
        if not old_dir.exists():
            continue
        
        print(f"Scanning {old_dir}...")
        
        # Find all .pt files
        for ckpt_file in old_dir.rglob("*.pt"):
            # Skip epoch checkpoints in subdirectories if we already have best
            # We'll migrate everything; index will deduplicate
            
            # Parse filename
            params = parse_checkpoint_filename(ckpt_file.name)
            if params['problem'] == 'unknown':
                print(f"  Skipping {ckpt_file.name} (unrecognized pattern)")
                skipped.append(ckpt_file)
                continue
            
            # Create experiment directory
            problem = params['problem']
            n_node = params.get('n_node', 'unknown')
            exp_name = f"{problem}_n{n_node}_migrated"
            
            # Add timestamp to make unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{exp_name}_{timestamp}"
            
            exp_dir = Path("experiments") / exp_name
            ckpt_dir = exp_dir / "checkpoints"
            
            if dry_run:
                print(f"  [DRY RUN] Would move {ckpt_file} → {ckpt_dir / ckpt_file.name}")
                migrated.append((ckpt_file, exp_dir))
                continue
            
            # Actually migrate
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            dest = ckpt_dir / ckpt_file.name
            shutil.copy2(ckpt_file, dest)  # copy2 preserves metadata
            
            # Create config.yaml
            config = generate_config_from_params(params)
            import yaml
            config_path = exp_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
            
            migrated.append((ckpt_file, exp_dir))
            print(f"  Migrated {ckpt_file.name} → {exp_dir}")
    
    # Generate report
    print("\n" + "="*60)
    print("MIGRATION REPORT")
    print("="*60)
    print(f"Total checkpoints found: {len(migrated) + len(skipped)}")
    print(f"Successfully migrated: {len(migrated)}")
    print(f"Skipped: {len(skipped)}")
    
    if dry_run:
        print("\nThis was a DRY RUN. No files were actually moved.")
        print("To execute migration, run with --execute flag.")
    else:
        print("\nMigration complete. Run `uv run scripts/indexExperiments.py --rebuild` to update index.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                       help="Actually move files (default is dry-run)")
    args = parser.parse_args()
    
    migrate_checkpoints(dry_run=not args.execute)


if __name__ == "__main__":
    main()
