# segmentation/segmentation_runner.py
import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
try:
    from .color_utils import get_color
except ImportError:
    from color_utils import get_color

def create_segmentation_video(seq_path: str,
                              output_path: str,
                              model_weights: str = "yolov8n-seg.pt",
                              device: str = "cpu",
                              conf: float = 0.25,
                              iou: float = 0.45,
                              mask_threshold: float = 0.5,
                              fps: int = 25,
                              alpha: float = 0.4):
    """
    Simple, working segmentation video creator with consistent colors
    """
    print(f"[segmentation_runner] Starting segmentation for {seq_path}")
    
    # Load model
    model = YOLO(model_weights)
    device_arg = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"[segmentation_runner] model={model_weights} device={device_arg}")

    # Get frames
    frames = sorted(glob.glob(os.path.join(seq_path, "img1", "*.jpg")))
    if not frames:
        print(f"No frames found in {seq_path}/img1")
        return

    print(f"Found {len(frames)} frames")

    # Load first frame to get dimensions
    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        print(f"Failed to load first frame: {frames[0]}")
        return
        
    h, w = first_frame.shape[:2]
    print(f"Frame dimensions: {w}x{h}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"Failed to open video writer: {output_path}")
        return
    
    print(f"Video writer opened: {output_path}")

    # Process frames
    for fidx, frame_path in enumerate(frames):
        if fidx % 10 == 0:
            print(f"Processing frame {fidx+1}/{len(frames)}")
            
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load frame: {frame_path}")
            continue

        # Ensure frame is correct size
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
            print(f"Resized frame {fidx+1} to {w}x{h}")

        # Run YOLO
        try:
            results = model(frame, conf=conf, iou=iou, device=device_arg)
            
            # Process results
            for res in results:
                # Get masks
                if hasattr(res, "masks") and res.masks is not None and res.masks.data is not None:
                    mask_data = res.masks.data.cpu().numpy()  # shape (N,H,W)
                    mask_data = np.clip(mask_data, 0, 1)
                    
                    for mi in range(mask_data.shape[0]):
                        mask = (mask_data[mi] > mask_threshold).astype(np.uint8) * 255
                        
                        # Resize mask to frame size if needed
                        if mask.shape[0] != h or mask.shape[1] != w:
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # Create colored overlay with consistent colors
                        color = get_color(mi)  # Use mask index for consistent coloring
                        colored = np.zeros_like(frame, dtype=np.uint8)
                        mask_bool = mask > 0
                        colored[mask_bool, 0] = color[0]  # B
                        colored[mask_bool, 1] = color[1]  # G  
                        colored[mask_bool, 2] = color[2]  # R
                        
                        # Apply overlay
                        frame = cv2.addWeighted(frame, 1.0, colored, alpha, 0)
                        
                        # Draw contour
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(frame, contours, -1, (0,0,0), 1)
                            cv2.drawContours(frame, contours, -1, color, 2)
                
        except Exception as e:
            print(f"Error processing frame {fidx+1}: {e}")
            continue

        # Write frame
        out.write(frame)

        # Clear cache occasionally
        if device_arg == "cuda" and fidx % 50 == 0:
            torch.cuda.empty_cache()

    # Close video writer
    out.release()
    print(f"[segmentation_runner] Video saved: {output_path}")
    
    # Check file size
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"Output file size: {size} bytes")
    else:
        print("ERROR: Output file not created!")