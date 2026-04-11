#!/usr/bin/env python3
"""
Optimized Crawl script for Rebuttal.
Produces a flattened CSV where each row is a unique configuration,
and columns represent performance across different test benchmarks.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

@dataclass
class ConfigPerformance:
    config_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    test_results: Dict[str, float] = field(default_factory=dict)
    train_metrics: Dict[str, Any] = field(default_factory=dict)

class RebuttalCrawler:
    def __init__(self, base_dir: str = "reviewer_experiments"):
        self.base_dir = Path(base_dir)
        # Search multiple directories for artifacts
        self.train_dirs = [
            self.base_dir / "results" / "artifacts" / "train",
            Path("results/artifacts/train")
        ]
        self.test_dirs = [
            self.base_dir / "results" / "artifacts" / "test",
            self.base_dir / "results" / "tmp",
            Path("results/artifacts/test"),
            Path("results/tmp")
        ]
        self.configs: Dict[str, ConfigPerformance] = {}

    def get_variant(self, name: str) -> str:
        if "nosmooth" in name: return "NoSmooth"
        if "anneal" in name: return "Anneal"
        if "static" in name: return "Static"
        if "lsglobal" in name: return "FullLS" if "fullls" in name else "GlobalLS"
        if "warmup" in name: return "Warmup"
        return "Standard"

    def parse_params(self, name: str) -> Dict[str, Any]:
        # tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06
        pattern = r'(\w+)_n(\d+)_k(\d+)_ants(\d+)_H(\d+)_miniH(\d+)_rho([\d.]+)_mne(\d+)'
        match = re.search(pattern, name)
        if match:
            return {
                'problem': match.group(1),
                'n_train': int(match.group(2)),
                'K': int(match.group(3)),
                'ants': int(match.group(4)),
                'H': int(match.group(5)),
                'miniH': int(match.group(6)),
                'S': int(match.group(5)) * int(match.group(6)),
                'rho': float(match.group(7)),
                'M': int(match.group(8)),
                'variant': self.get_variant(name)
            }
        return {'variant': self.get_variant(name)}

    def crawl(self):
        # 1. Process Training Artifacts (Configs and Training Stats)
        for t_dir in self.train_dirs:
            if not t_dir.exists(): continue
            for train_file in t_dir.glob("*_best.json"):
                config_name = train_file.name.replace("_best.json", "")
                if config_name in self.configs: continue
                
                try:
                    with open(train_file, 'r') as f:
                        data = json.load(f)
                except: continue
                
                params = self.parse_params(config_name)
                
                total_params = data.get('total_params')
                params_k = (total_params / 1000) if total_params is not None else 0
                
                total_train_time = data.get('total_train_time_s')
                train_time_h = (total_train_time / 3600) if total_train_time is not None else 0
                
                best_epoch = data.get('best_epoch') or {}
                val_gap = best_epoch.get('gap_pct', 0) if best_epoch else 0
                
                train_metrics = {
                    'mem_gb': data.get('peak_memory_gb') or 0,
                    'params_k': params_k,
                    'train_time_h': train_time_h,
                    'val_gap': val_gap
                }
                self.configs[config_name] = ConfigPerformance(config_name, params, {}, train_metrics)

        # 2. Process Test Artifacts
        test_files = []
        for d in self.test_dirs:
            if d.exists():
                test_files.extend(list(d.glob("*.json")))
        
        # Sort keys by length descending to match the most specific config first
        sorted_keys = sorted(self.configs.keys(), key=len, reverse=True)
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    data = json.load(f)
            except: continue

            # Strategy A: Check 'checkpoint' field in JSON (more reliable for CVRP)
            config_key = None
            checkpoint_path = data.get("checkpoint", "")
            if checkpoint_path:
                # Normalise path to ignore folder differences
                cp_name = Path(checkpoint_path).name.replace("_best.pt", "")
                for key in self.configs.keys():
                    if key == cp_name:
                        config_key = key
                        break
            
            # Strategy B: Filename matching (Fallback)
            if not config_key:
                name = test_file.name
                for key in sorted_keys:
                    if name.startswith(key):
                        config_key = key
                        break
            
            if not config_key: continue

            # Determine test scale and type
            # Standardize names: gap_synthetic_1000, gap_rldata_1000
            name = test_file.name.lower()
            test_scale = str(data.get("n_node", "unknown"))
            
            if "synthetic" in name:
                test_type = "synthetic"
            elif "rl_data" in name or "rldata" in name:
                test_type = "rldata"
            elif "rt0p4" in name:
                test_type = "rt0p4_synthetic"
            else:
                test_type = "test"
            
            # Special case for cross-scale tests in tmp folder
            if "test1000" in name: test_scale = "1000"
            if "test5000" in name: test_scale = "5000"

            col_name = f"gap_{test_type}_{test_scale}"
            
            # Extract gap_pct, mean_cost, and mean_time_s
            methods = data.get('methods', {})
            # Priority: model_anneal -> model_no_anneal -> model
            method_data = methods.get('model_anneal') or methods.get('model_no_anneal') or methods.get('model')
            
            if not method_data: continue
                
            gap = method_data.get('gap_pct')
            cost = method_data.get('mean_cost')
            time = method_data.get('mean_time_s')
            
            if gap is not None:
                self.configs[config_key].test_results[col_name] = gap
            if cost is not None:
                cost_col = f"cost_{test_type}_{test_scale}"
                self.configs[config_key].test_results[cost_col] = cost
            if time is not None:
                time_col = f"time_{test_type}_{test_scale}"
                self.configs[config_key].test_results[time_col] = time

    def to_csv(self, output_path: str):
        rows = []
        for config in self.configs.values():
            row = {**config.params, **config.train_metrics, **config.test_results}
            rows.append(row)
        
        df = pd.DataFrame(rows)
        # Reorder columns: Params first, then Train, then Test (Gap, Cost, and Time)
        param_cols = [c for c in df.columns if c in ['problem', 'n_train', 'K', 'ants', 'H', 'miniH', 'S', 'rho', 'M', 'variant']]
        train_cols = ['mem_gb', 'params_k', 'train_time_h', 'val_gap']
        
        test_cols = sorted([c for c in df.columns if (c.startswith("gap_") or c.startswith("cost_") or c.startswith("time_")) and c != "val_gap"])
        
        final_cols = param_cols + train_cols + test_cols
        # Filter to only existing columns
        final_cols = [c for c in final_cols if c in df.columns]
        df = df[final_cols]
        
        df.round(4).to_csv(output_path, index=False)
        print(f"Aggregated {len(df)} configs to {output_path}")
        print("\nColumn Overview:")
        for c in df.columns:
            print(f" - {c}")

if __name__ == "__main__":
    crawler = RebuttalCrawler()
    crawler.crawl()
    crawler.to_csv("rebuttal_results.csv")
