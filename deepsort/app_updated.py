import argparse
import cv2
import numpy as np
import glob
import os
import torch
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from tools.generate_detections import create_box_encoder
from detector import Detector
from torchreid.utils import FeatureExtractor

def read_sequence(seq_path):
    """Read image sequence from seq_path/img1/*.jpg."""
    frames = sorted(glob.glob(os.path.join(seq_path, "img1", "*.jpg")))
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is not None:
            yield frame, os.path.basename(frame_path)

def main(args):
    # Initialize detector
    detector = Detector(model_type=args.detector)
    
    # Initialize feature extractor
    if 'mars' in args.reid_model_path:
        encoder = create_box_encoder(args.reid_model_path, batch_size=32)
        reid_model = None
    else:
        model_name = 'resnet50_msmt17' if 'msmt17' in args.reid_model_path else 'resnet50_market1501'
        reid_model = FeatureExtractor(model_name=model_name, model_path=args.reid_model_path, device='cpu')
        encoder = None
    
    # Initialize tracker
    tracker = Tracker(metric="cosine", max_iou_distance=0.7, max_age=30, n_init=3)
    
    frame_generator = read_sequence(args.video_path)
    out_file = open(args.output_file, 'w')
    
    frame_idx = 1
    for frame, frame_name in frame_generator:
        # Detect objects
        detections = detector.detect(frame)
        if detections.size == 0:
            boxes, scores, names = [], [], []
        else:
            boxes = detections[:, :4]  # x1, y1, w, h
            scores = detections[:, 4]
            names = ['person'] * len(detections)
        
        # Extract features
        if encoder:
            features = encoder(frame, boxes)
        else:
            crops = [frame[int(y):int(y+h), int(x):int(x+w)] for x, y, w, h in boxes]
            features = reid_model(crops).cpu().numpy() if crops else np.array([])
        
        detections = [Detection(bbox, score, feature, name) for bbox, score, feature, name in zip(boxes, scores, features, names)]
        
        # Update tracker
        tracker.predict()
        tracker.update(detections)
        
        # Write tracks to file
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            x1, y1, x2, y2 = bbox
            out_file.write(f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\n")
        frame_idx += 1
    
    out_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to input video or image sequence directory")
    parser.add_argument("--output_file", required=True, help="Path to output tracking file")
    parser.add_argument("--detector", default="yolov8", choices=["yolov8", "yolov5", "detr"], help="Detector model")
    parser.add_argument("--reid_model_path", default="/Users/pk/Documents/2025/2q/HSE/DLCV/project/2/deep_sort/checkpoint/mars-small128.pb", help="Path to REID model")
    args = parser.parse_args()
    main(args)