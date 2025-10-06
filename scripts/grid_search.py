#!/usr/bin/env python3
"""
Grid search wrapper for run_full_tracking hyperparameter sweeps.
Assumes: motmetrics installed for HOTA eval; seq has gt/gt.txt.
Usage: python grid_search.py --sequence_dir /path/to/seq --detector fasterrcnn
       (or call grid_search() directly in Jupyter/IPython).
Updated for correct motmetrics HOTA computation (via compare_to_groundtruth_reweighting).
"""

import argparse
import os
import sys
import itertools
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import motmetrics as mm  # For HOTA eval; pip install motmetrics or git+https://github.com/cheind/py-motmetrics.git

# Add your project path (adjust if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_deepsort_full import run_full_tracking  # Your pipeline script (rename if needed)

def compute_hota(gt_file: str, pred_file: str, seq_name: str) -> float:
    """
    Compute HOTA from GT and pred MOT files using motmetrics (via reweighting).
    Returns: HOTA score (float).
    """
    try:
        # Load dataframes
        df_gt = mm.io.loadtxt(gt_file)
        df_test = mm.io.loadtxt(pred_file)
        
        # Compute reweighting for HOTA (thresholds for detection/association)
        th_list = np.arange(0.05, 0.99, 0.05)  # Standard MOTChallenge thresholds
        res_list = mm.utils.compare_to_groundtruth_reweighting(df_gt, df_test, "iou", distth=th_list)
        
        # Create metrics handler and compute HOTA (overall)
        mh = mm.metrics.create()
        summary = mh.compute_many(
            res_list,
            metrics=["deta_alpha", "assa_alpha", "hota_alpha"],
            generate_overall=True
        )
        
        # Extract overall HOTA
        hota = summary['hota_alpha'].iloc[-1] if 'hota_alpha' in summary.columns else 0.0
        print(f"  HOTA: {hota:.4f}")
        return float(hota)
    except Exception as e:
        print(f"  HOTA eval failed: {e}")
        return 0.0

def grid_search(sequence_dir: str,
                detector: str = "yolov5",
                sweeps: Dict[str, List[Any]] = None,
                fixed_args: Dict[str, Any] = None,
                output_dir: str = "grid_results") -> Dict[tuple, float]:
    """
    Run grid search over sweeps.
    
    Args:
        sequence_dir: Path to MOT sequence dir (e.g., with img1/, gt/gt.txt).
        detector: Detector name (e.g., 'fasterrcnn').
        sweeps: Dict of {param: [values]} to sweep (e.g., {'conf_threshold': [0.3, 0.4]}).
        fixed_args: Dict of fixed params (e.g., {'reid_model': 'osnet_x0_25'}).
        output_dir: Where to save per-run .txt and final CSV.
    
    Returns:
        Dict of {(param1_val, param2_val, ...): HOTA} for each combo.
    """
    if sweeps is None:
        sweeps = {}
    if fixed_args is None:
        fixed_args = {
            'reid_model': 'osnet_x0_25',
            'max_cosine_distance': 0.2,
            'device': 'cpu',
            'max_iou_distance': 0.7,  # Default from your code
            'max_age': 30,
            'n_init': 3,
            # Add more defaults as needed
        }
    
    # Grid combos
    param_names = list(sweeps.keys())
    combos = list(itertools.product(*(sweeps[name] for name in param_names)))
    
    results = {}
    gt_file = os.path.join(sequence_dir, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        raise ValueError(f"GT file not found: {gt_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    seq_name = os.path.basename(sequence_dir)
    
    print(f"Grid search: {len(combos)} combos for {seq_name} with {detector}")
    print(f"Sweeps: {sweeps}")
    
    for i, combo in enumerate(combos):
        # Merge fixed + sweep params
        run_args = dict(fixed_args)
        for name, val in zip(param_names, combo):
            run_args[name] = val
        
        # Unique output file
        suffix = "_".join(f"{n}{v}" for n, v in zip(param_names, combo))
        output_file = os.path.join(output_dir, f"{seq_name}_{detector}_{suffix}.txt")
        run_args['output_file'] = output_file
        
        print(f"\nRun {i+1}/{len(combos)}: {dict(zip(param_names, combo))}")
        
        # Run tracking
        try:
            run_results = run_full_tracking(
                sequence_dir=sequence_dir,
                detector_name=detector,
                **run_args  # Unpack all
            )
            print(f"  Saved: {output_file}")
        except Exception as e:
            print(f"  Run failed: {e}")
            continue
        
        # Eval HOTA
        hota = compute_hota(gt_file, output_file, seq_name)
        results[combo] = hota
        
        # Quick FPS proxy (if you add timing to run_full_tracking)
        # print(f"  FPS: {len(run_results) / total_time:.1f}")
    
    # Save summary CSV
    df = pd.DataFrame([
        {**dict(zip(param_names, combo)), 'HOTA': hota}
        for combo, hota in results.items()
    ])
    csv_path = os.path.join(output_dir, f"{seq_name}_{detector}_grid_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved: {csv_path}")
    print(df)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Grid search for MOT tracking")
    parser.add_argument("--sequence_dir", required=True, help="Path to sequence dir")
    parser.add_argument("--detector", default="yolov5", choices=["yolov5", "fasterrcnn", "retinanet"])
    parser.add_argument("--sweeps", nargs='+', help="Sweep format: param1:val1,val2 param2:val3,val4")
    parser.add_argument("--output_dir", default="grid_results")
    
    args = parser.parse_args()
    
    # Parse sweeps from CLI (e.g., --sweeps conf_threshold:0.3,0.4 max_age:15,30)
    sweeps = {}
    if args.sweeps:
        for s in args.sweeps:
            param, vals_str = s.split(':', 1)
            vals = [float(v) if '.' in v else int(v) for v in vals_str.split(',')]
            sweeps[param] = vals
    
    fixed_args = {}  # Override defaults here if needed
    
    grid_search(args.sequence_dir, args.detector, sweeps, fixed_args, args.output_dir)

if __name__ == "__main__":
    main()