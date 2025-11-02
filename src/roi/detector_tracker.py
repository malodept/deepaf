
# Optional YOLO wrapper. Falls back to whole-frame ROI if YOLO not available.
class OptionalYOLO:
    def __init__(self, weights="yolov8n.pt", conf=0.25):
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.conf = conf
            self.enabled = True
        except Exception as e:
            print("YOLO not available, fallback to full-frame ROI.", e)
            self.model = None; self.enabled = False
    def detect(self, frame):
        if not self.enabled:
            h,w = frame.shape[:2]
            return [(0,0,w,h,1.0,0)]
        res = self.model(frame, verbose=False, conf=self.conf)[0]
        out = []
        for b in res.boxes:
            x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0].item())
            cls = int(b.cls[0].item())
            out.append((x1,y1,x2-x1,y2-y1,conf,cls))
        return out
