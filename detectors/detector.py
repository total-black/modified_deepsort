import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import detection
from torchvision.ops import nms as tv_nms
from ultralytics import YOLO

class Detector:
    """
    Unified detector class for YOLOv5, Faster R-CNN, RetinaNet.
    Returns detections in [x1,y1,w,h,conf,cls] format.
    """

    def __init__(self, model_type: str = "yolov5", device: str = "cpu", verbose: bool = False):
        self.model_type = model_type.lower()
        self.device = device
        self.verbose = verbose

        if self.model_type == "yolov5":
            self.model = YOLO("yolov5s.pt")
            self.model.to(device)
        elif self.model_type == "fasterrcnn":
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.eval().to(device)
            self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        elif self.model_type == "retinanet":
            self.model = detection.retinanet_resnet50_fpn(pretrained=True)
            self.model.eval().to(device)
            self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        else:
            raise ValueError(f"Unsupported detector: {model_type}")

    @torch.no_grad()
    def detect(self, image: np.ndarray, conf_threshold: float = 0.3, nms_iou: float = 0.5, img_size: int = 640) -> np.ndarray:
        """
        Returns Nx6 array: [x1,y1,w,h,conf,cls]
        """
        try:
            if self.model_type == "yolov5":
                # ultralytics YOLO supports 'iou' and 'imgsz' parameters
                results = self.model(image, conf=conf_threshold, iou=nms_iou, imgsz=img_size)
                boxes = []
                for r in results:
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        conf = b.conf.item()
                        cls = int(b.cls.item())
                        # FIXED
                        boxes.append([x1, y1, x2 - x1, y2 - y1, conf, cls])
                if self.verbose:
                    print(f"[YOLOv5] Detected {len(boxes)} on frame")
                return np.array(boxes, dtype=np.float32)

            else:  # FRCNN / RetinaNet
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Resize so short side == img_size while preserving aspect ratio
                h, w = image_rgb.shape[:2]
                scale = 1.0
                if min(h, w) > 0 and img_size is not None:
                    scale = float(img_size) / float(min(h, w))
                    new_w, new_h = int(round(w * scale)), int(round(h * scale))
                    image_resized = cv2.resize(image_rgb, (new_w, new_h))
                else:
                    image_resized = image_rgb

                input_tensor = self.transform(image_resized).to(self.device)  # (C, H, W)
                predictions = self.model([input_tensor])[0]

                boxes, scores, labels = predictions['boxes'], predictions['scores'], predictions['labels']
                boxes, scores, labels = boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()

                # Map boxes back to original image scale if we resized
                try:
                    if scale != 1.0:
                        boxes = boxes.astype(np.float32) / float(scale)
                except Exception:
                    pass

                # apply torchvision NMS using provided IoU threshold
                if boxes.shape[0] > 0:
                    try:
                        boxes_tensor = torch.from_numpy(boxes).float().to(self.device)
                        scores_tensor = torch.from_numpy(scores).float().to(self.device)
                        keep = tv_nms(boxes_tensor, scores_tensor, float(nms_iou))
                        keep = keep.cpu().numpy()
                        boxes = boxes[keep]
                        scores = scores[keep]
                        labels = labels[keep]
                    except Exception:
                        # fallback: no NMS applied
                        pass

                dets = []
                for b, s, l in zip(boxes, scores, labels):
                    if l == 1 and s >= conf_threshold:  # only person
                        x1, y1, x2, y2 = b
                        dets.append([x1, y1, x2 - x1, y2 - y1, s, int(l)])
                if self.verbose:
                    unique_classes, class_counts = np.unique(labels, return_counts=True)
                    print(f"[{self.model_type.upper()}] All detections: {len(scores)} (max score: {np.max(scores):.3f})")
                    print(f"  Class breakdown: {dict(zip(unique_classes, class_counts))}")
                    print(f"[{self.model_type.upper()}] Detected {len(dets)} persons on frame")
                return np.array(dets, dtype=np.float32)

        except Exception as e:
            print(f"[{self.model_type} detect error]: {e}")
            return np.empty((0, 6), dtype=np.float32)