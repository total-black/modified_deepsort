
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import List, Optional

# Small helper
def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32)

class REIDManager:
    """
    Robust REID manager:
      - supports deep-person-reid FeatureExtractor (osnet),
      - supports FastReID if installed (best-effort),
      - supports generic torch model (transreid) if model object provided.
    Exposes:
      - extract_from_crops(crops: List[np.ndarray], batch_size=64, device='cuda') -> np.ndarray (N x D)
    Notes:
      - crops are RGB numpy uint8 arrays (H,W,3).
      - returned embeddings are L2-normalized float32.
    """

    def __init__(self, reid_type: str = "osnet", model_path: Optional[str] = None, device: str = "cpu"):
        self.reid_type = reid_type.lower()
        # Prefer requested device if available, else fall back
        req = (device or 'cpu')
        if req.startswith('cuda') and not torch.cuda.is_available():
            req = 'cpu'
        if req == 'mps' and not torch.backends.mps.is_available():
            req = 'cpu'
        self.device = req
        self.model = None
        self.transform = None
        self.feature_extractor_wrapper = None  # for deep-person-reid FeatureExtractor object if available
        self.feature_dim = None  # Will be set after first forward pass
        self._model_initialized = False

        # Default image transform (PIL-based) used by many REID models
        self.default_transform = transforms.Compose([
            transforms.Resize((256, 128)),  # height, width
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        if self.reid_type in ['osnet', 'osnet_x1_0']:
            self._init_osnet(model_path)
        elif self.reid_type == 'osnet_ibn':
            self._init_torchreid_model('osnet_ibn_x1_0', model_path)
        elif self.reid_type == 'osnet_ain':
            self._init_torchreid_model('osnet_ain_x1_0', model_path)
        elif self.reid_type == 'osnet_x0_25':
            self._init_torchreid_model('osnet_x0_25', model_path)
        elif self.reid_type == 'osnet_x0_5':
            self._init_torchreid_model('osnet_x0_5', model_path)
        elif self.reid_type == 'mobilenet':
            self._init_torchreid_model('mobilenetv2_x1_0', model_path)
        elif self.reid_type == 'resnet18':
            self._init_torchreid_model('resnet18', model_path)
        elif self.reid_type == 'resnet34':
            self._init_torchreid_model('resnet34', model_path)
        elif self.reid_type == 'resnet50':
            self._init_resnet50(model_path)
        elif self.reid_type == 'densenet121':
            self._init_torchreid_model('densenet121', model_path)
        else:
            raise ValueError(f"Unknown reid_type '{reid_type}'")

    # --------------------- initializers ---------------------

    def _init_osnet(self, model_path: Optional[str]):
        """
        Try to use deep-person-reid FeatureExtractor.
        API: FeatureExtractor(model_name, model_path, device) -> callable that returns features
        If not available, try to fall back to local torch model loading (best-effort).
        """
        try:
            # deep-person-reid's FeatureExtractor
            from torchreid.reid.utils import FeatureExtractor
            # model_path can be None; FeatureExtractor often downloads pretrained weights
            fe = FeatureExtractor(
                model_name='osnet_x1_0',
                model_path=model_path,   # can be None
                device=self.device
            )
            self.feature_extractor_wrapper = fe
            # we don't know exact dim; call once on dummy to get dim if necessary later
            self.representation_dim = None
            self.transform = self.default_transform
            print("[REID] OSNet (deep-person-reid FeatureExtractor) initialized")
        except Exception as e:
            print(f"[REID] OSNet init failed (FeatureExtractor): {e}")
            # fall back: try torchreid older interfaces or raise
            # keep model None and let extract_from_crops attempt a generic torch path
            self.feature_extractor_wrapper = None
            self.transform = self.default_transform
            raise
    
    def _init_resnet50(self, model_path: Optional[str]):
        """
        Initialize ResNet50 REID from deep-person-reid.
        Third REID model for diversity.
        """
        try:
            from torchreid.reid.utils import FeatureExtractor
            fe = FeatureExtractor(
                model_name='resnet50',
                model_path=model_path,
                device=self.device
            )
            self.feature_extractor_wrapper = fe
            self.representation_dim = None
            self.transform = self.default_transform
            print("[REID] ResNet50 (deep-person-reid FeatureExtractor) initialized")
        except Exception as e:
            print(f"[REID] ResNet50 init failed: {e}")
            self.feature_extractor_wrapper = None
            self.transform = self.default_transform
            raise
    
    def _init_torchreid_model(self, model_name: str, model_path: Optional[str]):
        """
        Generic initializer for any torchreid FeatureExtractor model.
        """
        try:
            from torchreid.reid.utils import FeatureExtractor
            fe = FeatureExtractor(
                model_name=model_name,
                model_path=model_path,
                device=self.device
            )
            self.feature_extractor_wrapper = fe
            self.representation_dim = None
            self.transform = self.default_transform
            print(f"[REID] {model_name} (deep-person-reid FeatureExtractor) initialized")
        except Exception as e:
            print(f"[REID] {model_name} init failed: {e}")
            self.feature_extractor_wrapper = None
            self.transform = self.default_transform
            raise


    # --------------------- extraction helpers ---------------------
    def get_feature_dim(self) -> int:
        """
        Get feature dimension, initializing model if necessary.
        
        Returns:
            Feature dimension
        """
        if self.feature_dim is None:
            # Initialize with dummy data to get feature dimension
            dummy_crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
            dummy_features = self.extract_from_crops([dummy_crop], batch_size=1, device=self.device)
            self.feature_dim = dummy_features.shape[1]
        
        return self.feature_dim

    def extract_from_crops(self, crops: List[np.ndarray], batch_size: int = 64, device: Optional[str] = None, use_fp16: bool = False) -> np.ndarray:
        """
        Public API: extracts embeddings from list of RGB numpy crops.
        Returns: (N, D) float32 L2-normalized embeddings.
        """
        if device is None:
            device = self.device
        if len(crops) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # FeatureExtractor wrapper (deep-person-reid)
        if self.feature_extractor_wrapper is not None:
            embeddings = self._extract_via_feature_extractor(crops, batch_size, device)
        # Generic torch model path: try to use self.model if available
        elif self.model is not None:
            embeddings = self._extract_via_torch_model(crops, batch_size, device, use_fp16)
        else:
            raise RuntimeError("No REID backend is available. Please install/run a supported REID repo or provide a torchscript model.")
        
        # Set feature dimension if not set
        if self.feature_dim is None and len(embeddings) > 0:
            self.feature_dim = embeddings.shape[1]
        
        return embeddings


    def _extract_via_feature_extractor(self, crops: List[np.ndarray], batch_size: int, device: str) -> np.ndarray:
        """
        deep-person-reid FeatureExtractor accepts list of numpy arrays or paths.
        """
        feats_all = []
        try:
            # FeatureExtractor expects numpy arrays (RGB, uint8, shape (H, W, 3))
            # Call the wrapper - it should handle batching internally
            features = self.feature_extractor_wrapper(crops)
            # Convert to numpy
            if isinstance(features, torch.Tensor):
                feats_all = features.cpu().numpy().astype(np.float32)
            elif isinstance(features, np.ndarray):
                feats_all = features.astype(np.float32)
            else:
                # Try to convert whatever it is
                feats_all = np.asarray(features, dtype=np.float32)
        except Exception as e:
            print(f"[REID] FeatureExtractor call failed: {e}")
            raise

        if len(feats_all) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize_np(feats_all)

    def _extract_via_torch_model(self, crops: List[np.ndarray], batch_size: int, device: str, use_fp16: bool = False) -> np.ndarray:
        """
        Generic torch model forward. Assumes self.model is a torch.nn.Module that accepts a batch tensor.
        """
        tf = self.transform or self.default_transform
        tensors = []
        for crop in crops:
            pil = Image.fromarray(crop)
            t = tf(pil)
            tensors.append(t)
        all_feats = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size], dim=0).to(device)
            if use_fp16:
                batch = batch.half()
                self.model.half()
            with torch.no_grad():
                out = self.model(batch)
                # normalize handling of outputs
                if isinstance(out, torch.Tensor):
                    feats = out.cpu().numpy()
                elif isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor):
                    feats = out[0].cpu().numpy()
                elif isinstance(out, (list, tuple, np.ndarray)):
                    feats = np.asarray(out)
                elif isinstance(out, dict) and 'features' in out:
                    feats = out['features']
                    if isinstance(feats, torch.Tensor):
                        feats = feats.cpu().numpy()
                else:
                    # last resort: try to convert to numpy
                    try:
                        feats = np.asarray(out)
                    except Exception:
                        raise RuntimeError("Unsupported model output format; please wrap model to return tensor or np.ndarray")
            all_feats.append(feats.astype(np.float32))
        if len(all_feats) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        feats = np.vstack(all_feats)
        return l2_normalize_np(feats)

    def get_model_info(self):
        return {
            'type': self.reid_type,
            'device': self.device,
            'has_feature_extractor_wrapper': self.feature_extractor_wrapper is not None,
            'has_model': self.model is not None
        }

# Factory convenience
def create_reid_manager(reid_type='osnet', model_path=None, device='cpu'):
    return REIDManager(reid_type=reid_type, model_path=model_path, device=device)

