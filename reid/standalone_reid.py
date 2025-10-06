#!/usr/bin/env python3
"""
Pure Appearance-Only Tracker (No Kalman, No IoU)

This tracker uses ONLY REID features for association.
Implements the method suggested in the assignment:
- Hungarian assignment based on cosine similarity of embeddings
- Track expiry after max_age frames without matches
- No motion model, no IoU - pure appearance
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2  # For potential padding, but not used here

class StandaloneREID:
    """
    Tracks objects using only appearance features (REID embeddings).
    
    No Kalman filter, no IoU - just appearance matching.
    This properly evaluates REID quality.
    """
    
    def __init__(self, appearance_thresh=0.2, max_age=30):
        """
        Args:
            appearance_thresh: Minimum cosine similarity for matching (0-1)
            max_age: Maximum frames without match before track is deleted
        """
        self.appearance_thresh = appearance_thresh
        self.max_age = max_age
        
        self.tracks = {}  # {track_id: {'feature': np.array, 'bbox': [x,y,w,h], 'age': int, 'last_frame': int}}
        self.next_id = 1
        self.current_frame = 0
    
    def update(self, detections, features):
        """
        Update tracker with new detections and their features.
        
        Args:
            detections: List of bboxes [[x,y,w,h], ...] (xywh)
            features: np.array (N, D) L2-normalized features
        
        Returns:
            assignments: List of (detection_idx, track_id) tuples
        """
        self.current_frame += 1
        
        if len(detections) == 0:
            # Age all tracks
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
            return []
        
        # Ensure features are L2-normalized
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms
        
        if len(self.tracks) == 0:
            # No existing tracks - create new ones for all detections
            assignments = []
            for i, (bbox, feat) in enumerate(zip(detections, features)):
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'feature': feat.copy(),
                    'bbox': [int(round(b)) for b in bbox],  # FIXED: Round for MOT
                    'age': 0,
                    'last_frame': self.current_frame
                }
                assignments.append((i, track_id))
            return assignments
        
        # Build cost matrix (1 - cosine similarity)
        track_ids = list(self.tracks.keys())
        track_features = np.vstack([self.tracks[tid]['feature'] for tid in track_ids])  # (M, D)
        
        # Cosine similarity matrix (since both are L2-normalized)
        similarities = track_features.dot(features.T)  # (M, N)
        cost_matrix = 1.0 - similarities  # Convert to cost
        
        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matched_tracks = set()
        matched_dets = set()
        assignments = []
        
        for row, col in zip(row_indices, col_indices):
            if similarities[row, col] >= self.appearance_thresh:
                track_id = track_ids[row]
                # Update track
                self.tracks[track_id]['feature'] = features[col].copy()  # Update with new feature
                self.tracks[track_id]['bbox'] = [int(round(b)) for b in detections[col]]  # FIXED: Round
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['last_frame'] = self.current_frame
                
                matched_tracks.add(track_id)
                matched_dets.add(col)
                assignments.append((col, track_id))
        
        # Unmatched tracks - age them
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
        
        # Unmatched detections - create new tracks
        for i in range(len(detections)):
            if i not in matched_dets:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'feature': features[i].copy(),
                    'bbox': [int(round(b)) for b in detections[i]],  # FIXED: Round
                    'age': 0,
                    'last_frame': self.current_frame
                }
                assignments.append((i, track_id))
        
        return assignments
    
    def get_active_tracks(self):
        """
        Get all tracks that were matched in the current frame.
        
        Returns:
            List of (track_id, bbox) tuples for MOT output
        """
        active = []
        for tid, track_data in self.tracks.items():
            if track_data['last_frame'] == self.current_frame:
                active.append((tid, track_data['bbox']))
        return active