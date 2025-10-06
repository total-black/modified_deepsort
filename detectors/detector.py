import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import detection
from ultralytics import YOLO

class Detector:
    """
    Unified detector class for YOLOv5, Faster R-CNN, RetinaNet.
    Returns detections in [x1,y1,w,h,conf,cls] format.
    """

    def __init__(self, model_type: str = "yolov5", device: str = "cpu"):
        self.model_type = model_type.lower()
        self.device = device

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
    def detect(self, image: np.ndarray, conf_threshold: float = 0.3) -> np.ndarray:
        """
        Returns Nx6 array: [x1,y1,w,h,conf,cls]
        """
        try:
            if self.model_type == "yolov5":
                results = self.model(image, conf=conf_threshold)
                boxes = []
                for r in results:
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        conf = b.conf.item()
                        cls = int(b.cls.item())
                        # FIXED
                        boxes.append([x1, y1, x2 - x1, y2 - y1, conf, cls])
                print(f"[YOLOv5] Detected {len(boxes)} on frame")
                return np.array(boxes, dtype=np.float32)

            else:  # FRCNN / RetinaNet
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # FIXED 
                input_tensor = self.transform(image_rgb).to(self.device)  # (C, H, W)
                predictions = self.model([input_tensor])[0]

                boxes, scores, labels = predictions['boxes'], predictions['scores'], predictions['labels']
                boxes, scores, labels = boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()

                # DEBUG: Log all detections (pre-filter) to check if model sees anything
                unique_classes, class_counts = np.unique(labels, return_counts=True)
                print(f"[{self.model_type.upper()}] All detections: {len(scores)} (max score: {np.max(scores):.3f})")
                print(f"  Class breakdown: {dict(zip(unique_classes, class_counts))}")
                
                dets = []
                for b, s, l in zip(boxes, scores, labels):
                    if l == 1 and s >= conf_threshold:  # only person
                        x1, y1, x2, y2 = b
                        dets.append([x1, y1, x2 - x1, y2 - y1, s, int(l)])
                print(f"[{self.model_type.upper()}] Detected {len(dets)} persons on frame")
                return np.array(dets, dtype=np.float32)

        except Exception as e:
            print(f"[{self.model_type} detect error]: {e}")
            return np.empty((0, 6), dtype=np.float32)