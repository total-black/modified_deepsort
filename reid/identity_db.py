import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, deque
import threading

class IdentityDB:
    def __init__(self, knn_k=3, radius=0.6, max_descriptors_per_id=200):
        self.knn_k = knn_k
        self.radius = radius
        self.max_descriptors_per_id = max_descriptors_per_id

        self._id_to_descriptors = defaultdict(list)   # id -> list of np arrays
        self._id_to_centroid = {}                     # id -> np.array
        self._next_id = 1
        self._lock = threading.Lock()

        # flat index cache
        self._rebuild_index()

    def _rebuild_index(self):
        # Build flat descriptors array and mapping
        self._flat_descs = []
        self._flat_id_map = []
        for id_, descs in self._id_to_descriptors.items():
            for d in descs:
                self._flat_descs.append(d)
                self._flat_id_map.append(id_)
        if len(self._flat_descs) > 0:
            self._flat_descs = np.vstack(self._flat_descs)
            self._nn = NearestNeighbors(n_neighbors=min(self.knn_k, len(self._flat_descs)), metric='cosine')
            self._nn.fit(self._flat_descs)
        else:
            self._flat_descs = None
            self._nn = None

    def find_identity(self, descriptor: np.ndarray):
        """
        Returns (identity_id, score) or (None, None)
        score: mean cosine similarity across k neighbors (higher is better)
        """
        if self._nn is None:
            return None, None

        descriptor = descriptor.reshape(1, -1)
        dists, idxs = self._nn.kneighbors(descriptor, return_distance=True)
        # sklearn NearestNeighbors with metric='cosine' returns distances in [0,2] (cosine dist)
        # convert to similarity: sim = 1 - dist
        sims = 1.0 - dists[0]
        ids = [self._flat_id_map[i] for i in idxs[0]]
        # aggregate per identity: take max sim per unique id
        best = {}
        for _id, s in zip(ids, sims):
            if _id not in best or s > best[_id]:
                best[_id] = s
        # pick best identity and score
        best_id, best_score = max(best.items(), key=lambda x: x[1])
        if best_score >= (1.0 - self.radius):  # radius is cosine distance threshold; tweak semantics
            return best_id, float(best_score)
        return None, None

    def create_identity(self, descriptor: np.ndarray, metadata=None):
        with self._lock:
            new_id = self._next_id
            self._next_id += 1
            self._id_to_descriptors[new_id].append(descriptor.copy())
            self._id_to_centroid[new_id] = np.mean(self._id_to_descriptors[new_id], axis=0)
            self._rebuild_index()
            return new_id

    def update_identity(self, identity_id: int, descriptor: np.ndarray):
        with self._lock:
            arr = self._id_to_descriptors[identity_id]
            arr.append(descriptor.copy())
            if len(arr) > self.max_descriptors_per_id:
                arr.pop(0)
            self._id_to_centroid[identity_id] = np.mean(arr, axis=0)
            self._rebuild_index()

    def get_centroid(self, identity_id):
        return self._id_to_centroid.get(identity_id, None)

    def resolve_id_conflicts(self, assignments):
        inv = defaultdict(list)
        for t, val in assignments.items():
            if val is None: continue
            ident_id, score = val  # Unpack (id, score)
            inv[ident_id].append((t, score))  # FIXED: Per-identity_id, store (track_id, score) tuples

        for ident, track_scores in inv.items():
            if len(track_scores) > 1:
                # FIXED: Keep top 2 by score (for close persons)
                sorted_ts = sorted(track_scores, key=lambda x: x[1], reverse=True)[:2]
                keep_ts = {ts[0] for ts in sorted_ts}
                for t, s in track_scores:
                    if t not in keep_ts:
                        assignments[t] = None
        return assignments