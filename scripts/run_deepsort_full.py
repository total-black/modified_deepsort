"""
Run DeepSORT with Full Pipeline

"""

import argparse
import os
import sys
import numpy as np
import cv2
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modified_deepsort.tracker import ModifiedTracker
from detectors.detector import Detector

def sanitize_detections(detections, image_shape):
    """
    Convert various detector outputs to DeepSORT-friendly format:
    - Ensures [x1,y1,x2,y2,score]
    - Fixes negative or inverted boxes
    - Clips to image bounds
    - Filters out boxes with zero/negative width or height
    """
    h_img, w_img = image_shape[:2]
    
    clean_dets = []
    for det in detections:
        if len(det) < 5:
            continue
            
        x1, y1, x2, y2, score = det[:5]
        
        # Fix inverted coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Clip to image size
        x1 = np.clip(x1, 0, w_img - 1)
        y1 = np.clip(y1, 0, h_img - 1)
        x2 = np.clip(x2, 0, w_img - 1)
        y2 = np.clip(y2, 0, h_img - 1)
        
        # Filter invalid boxes
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue
            
        clean_dets.append([x1, y1, x2, y2, score])
    
    return np.array(clean_dets, dtype=np.float32)


def run_full_tracking(sequence_dir: str,
                     detector_name: str = "yolov5",
                     reid_model: str = "osnet_x0_25",
                     max_cosine_distance: float = 0.2,
                     conf_threshold: float = 0.5,
                     device: str = "cpu",
                     output_file: str = None) -> List[List[float]]:
    """
    Run full DeepSORT tracking pipeline with detector + REID.
    
    Args:
        sequence_dir: Path to sequence directory
        detector_name: Detector model to use
        reid_model: REID model to use
        max_cosine_distance: Maximum cosine distance for REID matching
        conf_threshold: Detection confidence threshold
        device: Device to run on
        output_file: Output file path
        
    Returns:
        List of tracking results in MOT format
    """
    detector = Detector(detector_name, device=device, verbose=False)
    tracker = ModifiedTracker(
        reid_model=reid_model,
        max_cosine_distance=max_cosine_distance,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        device=device,
        verbose=False
    )
    
    img_dir = os.path.join(sequence_dir, "img1")
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        return []
    
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    results = []
    
    for frame_idx, frame_file in enumerate(frame_files):
        frame_id = frame_idx + 1
        image = cv2.imread(os.path.join(img_dir, frame_file))
        if image is None:
            continue

        raw_dets = detector.detect(image, conf_threshold=conf_threshold)

        # FIXED: Mixed class filter (YOLO=0, torchvision=1)
        person_class = 0 if detector_name.startswith("yolo") else 1
        person_dets = [det for det in raw_dets if len(det) >= 6 and det[5] == person_class]
        raw_dets = [det[:5] for det in person_dets]  # Drop class ID

        converted = []
        for det in raw_dets:
            xc, yc, w, h, score = det[:5]
            x1 = xc - w/2
            y1 = yc - h/2
            x2 = xc + w/2
            y2 = yc + h/2
            converted.append([x1, y1, x2, y2, score])
        raw_dets = converted

        detections = sanitize_detections(raw_dets, image.shape)

        if len(detections) > 0:
            # Convert xyxy to xywh for tracker
            x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
            scores = detections[:, 4]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            boxes_xywh = np.column_stack([x_center, y_center, width, height])

            # FIXED: Integrated update with identity
            tracker.update_with_image(image, boxes_xywh, scores, frame_id)
        else:
            # Empty frame: Predict to advance Kalman
            tracker.tracker.predict()

        # FIXED: Always get dict tracks (filters confirmed, adds 'identity')
        tracks = tracker.get_tracks()

        # Clean frame summary
        n_dets = len(detections)
        n_tracks = len(tracks)
        n_with_id = sum(1 for t in tracks if t.get('identity') is not None)
        print(f"Frame {frame_id}: {n_dets} feats, {n_tracks} mapped, {n_with_id}/{n_tracks} voted ID")

        for track in tracks:
            x1, y1, x2, y2 = track['bbox']  # Safe: dict from get_tracks()
            w, h = x2 - x1, y2 - y1
            # FIXED: Round bbox to integers for MOT consistency with GT
            x1, y1, w, h = int(round(x1)), int(round(y1)), int(round(w)), int(round(h))
            results.append([frame_id, track['id'], x1, y1, w, h, track['score'], -1, -1, -1])
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for res in results:
                f.write(','.join(map(str, res)) + '\n')
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full DeepSORT pipeline")
    parser.add_argument("--sequence_dir", required=True, help="Path to sequence directory")
    parser.add_argument("--detector", default="yolov5", 
                       choices=["yolov5", "fasterrcnn", "retinanet"],
                       help="Detector model to use")
    parser.add_argument("--reid_model", default="osnet_x0_25", 
                       choices=["mars", "osnet_x0_25", "resnet50", "densenet121"],
                       help="REID model to use")
    parser.add_argument("--max_cosine_distance", type=float, default=0.2,
                       help="Maximum cosine distance for REID matching")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                       help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                       help="Device to run on")
    parser.add_argument("--output_file", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.output_file:
        seq_name = os.path.basename(args.sequence_dir)
        args.output_file = f"results/{seq_name}_full_{args.detector}_{args.reid_model}.txt"
    
    run_full_tracking(
        sequence_dir=args.sequence_dir,
        detector_name=args.detector,
        reid_model=args.reid_model,
        max_cosine_distance=args.max_cosine_distance,
        conf_threshold=args.conf_threshold,
        device=args.device,
        output_file=args.output_file
    )
    
    print(f"Full tracking completed. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()