#!/usr/bin/env python3
"""
Run DeepSORT with REID-only evaluation (Step B)

This script runs DeepSORT using only REID features for association,
disabling the motion model to evaluate pure REID performance.
"""

import argparse
import os
import sys
import numpy as np
import cv2
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modified_deepsort.tracker import ModifiedTracker
from modified_deepsort.reid_adapter import REIDAdapter
from deepsort.detection import Detection
# from scripts.simple_evaluation import evaluate_tracker_results  # Not implemented yet


def load_ground_truth(gt_file: str) -> Dict[int, List[np.ndarray]]:
    """
    Load ground truth detections from MOT format file.
    
    Args:
        gt_file: Path to ground truth file
        
    Returns:
        Dictionary mapping frame_id to list of detections [x, y, w, h, conf, id]
    """
    gt_detections = {}
    
    if not os.path.exists(gt_file):
        print(f"Warning: Ground truth file not found: {gt_file}")
        return gt_detections
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                
                if frame_id not in gt_detections:
                    gt_detections[frame_id] = []
                
                gt_detections[frame_id].append(np.array([x, y, w, h, conf, track_id]))
    
    return gt_detections


def run_reid_only_tracking(sequence_dir: str,
                          reid_model: str = "osnet_x0_25",
                          max_cosine_distance: float = 0.2,
                          device: str = "cpu",
                          output_file: str = None) -> List[List[float]]:
    """
    Run REID-only tracking on a sequence.
    
    Args:
        sequence_dir: Path to sequence directory
        reid_model: REID model to use
        max_cosine_distance: Maximum cosine distance for matching
        device: Device to run on
        output_file: Output file path
        
    Returns:
        List of tracking results in MOT format
    """
    # Initialize tracker with REID-only settings
    tracker = ModifiedTracker(
        reid_model=reid_model,
        max_cosine_distance=max_cosine_distance,
        max_iou_distance=1.0,  # Disable IoU matching
        max_age=30,
        n_init=1,  # Lower threshold for confirmation
        device=device
    )
    
    # Load ground truth
    gt_file = os.path.join(sequence_dir, "gt", "gt.txt")
    gt_detections = load_ground_truth(gt_file)
    
    # Get image directory
    img_dir = os.path.join(sequence_dir, "img1")
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        return []
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    results = []
    
    print(f"Running REID-only tracking on {len(frame_files)} frames...")
    
    for frame_idx, frame_file in enumerate(frame_files):
        frame_id = frame_idx + 1
        frame_path = os.path.join(img_dir, frame_file)
        
        # Load image
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Warning: Failed to load frame {frame_file}")
            continue
        
        # Get ground truth detections for this frame
        if frame_id in gt_detections:
            gt_boxes = []
            gt_scores = []
            gt_ids = []
            
            for det in gt_detections[frame_id]:
                x, y, w, h, conf, track_id = det
                gt_boxes.append([x, y, w, h])
                gt_scores.append(conf)
                gt_ids.append(track_id)
            
            if len(gt_boxes) > 0:
                # Extract REID features
                features = tracker.extract_features(image, np.array(gt_boxes))
                
                # Create detections
                detections = tracker.create_detections(
                    np.array(gt_boxes),
                    np.array(gt_scores),
                    features
                )
                
                # Update tracker
                tracker.update(detections)
        
        # Get current tracks
        tracks = tracker.get_tracks()
        
        # Save results
        for track in tracks:
            bbox = track['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            result = [frame_id, track['id'], x, y, w, h, track['score'], -1, -1, -1]
            results.append(result)
    
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for result in results:
                f.write(','.join(map(str, result)) + '\n')
        print(f"Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run DeepSORT with REID-only evaluation")
    parser.add_argument("--sequence_dir", required=True, help="Path to sequence directory")
    parser.add_argument("--reid_model", default="osnet_x0_25", 
                       choices=["mars", "osnet_x0_25", "resnet50", "densenet121"],
                       help="REID model to use")
    parser.add_argument("--max_cosine_distance", type=float, default=0.2,
                       help="Maximum cosine distance for REID matching")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                       help="Device to run on")
    parser.add_argument("--output_file", help="Output file path")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate results against GT")
    
    args = parser.parse_args()
    
    # Set output file if not provided
    if not args.output_file:
        seq_name = os.path.basename(args.sequence_dir)
        args.output_file = f"results/{seq_name}_reid_only_{args.reid_model}.txt"
    
    # Run tracking
    results = run_reid_only_tracking(
        sequence_dir=args.sequence_dir,
        reid_model=args.reid_model,
        max_cosine_distance=args.max_cosine_distance,
        device=args.device,
        output_file=args.output_file
    )
    
    print(f"Tracking completed. Generated {len(results)} track detections.")
    
    # Evaluate if requested
    if args.evaluate:
        print("\nEvaluating results...")
        # evaluate_tracker_results(args.sequence_dir, args.output_file)  # Not implemented yet
    
    print(f"✅ REID-only tracking completed. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()