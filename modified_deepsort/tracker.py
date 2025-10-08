import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from deepsort.tracker import Tracker as OriginalTracker
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.detection import Detection
from reid.reid_manager import create_reid_manager
from reid.identity_db import IdentityDB

class ModifiedTracker:
    def __init__(self,
                 reid_model: str = "mars",
                 max_cosine_distance: float = 0.2,
                 nn_budget: int = 100,
                 max_iou_distance: float = 0.7,
                 max_age: int = 30,
                 n_init: int = 3,
                 device: str = "cpu",
                 identity_knn_k: int = 3,
                 identity_cosine_threshold: float = 0.5,
                 identity_window_frames: int = 100,
                 create_identity_on_appearance: bool = True,
                 verbose: bool = False):        
        """
        Identity-aware ModifiedTracker

        identity_cosine_threshold: required cosine SIMILARITY (not distance) threshold
                                  if using NearestNeighbors(cosine), convert appropriately.
        identity_window_frames: sliding window (in frames) used for majority voting
        """
        self.reid_model = reid_model
        self.device = device
        self.verbose = verbose


        # REID manager & feature dim
        self.reid_manager = create_reid_manager(reid_model, device=device)
        self.feature_dim = self.reid_manager.get_feature_dim()

        # distance metric and underlying DeepSORT tracker
        self.metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = OriginalTracker(self.metric,
                                       max_iou_distance=max_iou_distance,
                                       max_age=max_age,
                                       n_init=n_init)

        # Identity DB (standalone REID database)
        # `radius` below is interpreted as cosine distance threshold; we'll convert usage to similarity.
        self.identity_db = IdentityDB(knn_k=identity_knn_k, radius=(1.0 - identity_cosine_threshold))
        self.identity_cosine_threshold = identity_cosine_threshold
        self.identity_window_frames = identity_window_frames
        self.create_identity_on_appearance = create_identity_on_appearance

        # per-track history: track_id -> deque of (frame_id, identity_id, score)
        self.track_identity_history = defaultdict(lambda: deque(maxlen=self.identity_window_frames))

        if self.verbose:
            print(f"ModifiedTracker(identity_db knn_k={identity_knn_k}, cos_thr={identity_cosine_threshold}) initialized")
    # ----------------------- existing functions (unchanged) -----------------------
    def update(self, detections: List[Detection]) -> List:
        """ Update underlying tracker with Detection objects and return internal tracks """
        self.tracker.predict()
        self.tracker.update(detections)
        return self.tracker.tracks

    def extract_features(self, image: np.ndarray, boxes: List[np.ndarray]) -> np.ndarray:
        crops = []
        for box in boxes:
            x, y, w, h = box.astype(int)
            if w <= 0 or h <= 0:
                crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
            else:
                crop_bgr = image[y:y+h, x:x+w]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)  # FIXED: To RGB
                crops.append(crop_rgb)
        features = self.reid_manager.extract_from_crops(crops, batch_size=16)
        # L2 normalize (already done in manager, but safe)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms
        return features

    def create_detections(self,
                          boxes: np.ndarray,
                          scores: np.ndarray,
                          features: np.ndarray) -> List[Detection]:
        detections = []
        for i in range(len(boxes)):
            detections.append(Detection(boxes[i], scores[i], features[i]))
        return detections

    def get_tracks(self) -> List[Dict]:
        """Return simplified track dicts for external use"""
        tracks = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            bbox = t.to_tlbr()
            tracks.append({
                'id': t.track_id,
                'bbox': bbox,
                'score': getattr(t, 'score', 0.0),
                'age': getattr(t, 'age', 0),
                'time_since_update': getattr(t, 'time_since_update', 0),
                'state': getattr(t, 'state', None),
                'identity': getattr(t, 'identity', None)  # may be set by identity assignment
            })
        return tracks

    # ----------------------- identity integration logic -----------------------
    def update_with_image(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, frame_id: int) -> List:
        """
        Run the whole detection->features->update->identity assignment pipeline for one frame.
        Returns underlying tracks after identity assignment.
        """
        # 1) Extract features for all detection boxes
        features = self.extract_features(image, boxes)

        # 2) Create detection objects and update tracker
        detections = self.create_detections(boxes, scores, features)
        tracks = self.update(detections)  # runs predict() + update()

        # 3) Map detections -> track_id using IoU or builtin association (we'll match by centers)
        det_to_track = self._create_detection_to_track_mapping(tracks, boxes)
        if self.verbose:
            print(f"Frame {frame_id}: {len(features)} feats, {len(det_to_track)} mapped (IoU>0.1)")
        
        # 4) For each detection descriptor, find or create identity and append to per-track history
        for det_idx, feat in enumerate(features):
            track_id = det_to_track.get(det_idx, None)
            # find identity by querying identity_db
            identity_id, score = self.identity_db.find_identity(feat)
            if identity_id is None and self.create_identity_on_appearance:
                # create new identity and assign it
                identity_id = self.identity_db.create_identity(feat, metadata={'first_frame': frame_id})
                score = 1.0  # newly created => high confidence
        # update identity DB with descriptor if identity found (keeps centroids up-to-date)
        if identity_id is not None:
            self.identity_db.update_identity(identity_id, feat)
        # store candidate into track history
        if track_id is not None:
            append_score = float(score) if score is not None else 1.0
            self.track_identity_history[track_id].append((frame_id, identity_id, append_score))

        # 5) Decide per-track chosen identity via window majority & basic tie-breaking
        chosen_assignments = {}
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            tid = t.track_id
            chosen_id, vote_weight = self.pick_identity_for_track(tid, current_frame=frame_id)
            if chosen_id is not None:
                chosen_assignments[tid] = (chosen_id, vote_weight)  # FIXED: Tuple (ID, score)
            else:
                chosen_assignments[tid] = None
        non_none = sum(1 for v in chosen_assignments.values() if v is not None)
        if self.verbose:
            print(f"  {non_none}/{len(chosen_assignments)} tracks voted ID")

        # 6) Resolve conflicts across tracks (simple policy) and apply final identities back to track objects
        resolved = self.identity_db.resolve_id_conflicts(chosen_assignments.copy())
        # apply resolved identities to actual track objects
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            final_val = resolved.get(t.track_id, None)
            if final_val is not None and isinstance(final_val, tuple):
                final_identity = final_val[0]  # FIXED: Extract ID from tuple
            else:
                final_identity = final_val
            # attach identity to track (so rest of pipeline/visualiser can use it)
            setattr(t, 'identity', final_identity)

        for t in tracks[:5]:
            if self.verbose:
                print(f"  Track {t.track_id} -> Identity {getattr(t,'identity',None)}")
    
        return tracks