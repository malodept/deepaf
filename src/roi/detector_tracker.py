import numpy as np
# Optional YOLO wrapper. Falls back to whole-frame ROI if YOLO not available.

class OptionalYOLO:
    def __init__(self, weights="yolov8m.pt", conf=0.25, cls_keep=(14,), ema=0.6):
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.conf = conf
            self.enabled = True
            self.cls_keep = set(cls_keep) if cls_keep is not None else None
            self._ema = ema
            self._prev = None
        except Exception as e:
            print("YOLO not available, fallback to full-frame ROI.", e)
            self.model = None; self.enabled = False
    def detect(self, frame):
        if not self.enabled:
            h,w = frame.shape[:2]
            return [(0,0,w,h,1.0,0)]
        res = self.model(frame, verbose=False, conf=self.conf, iou=0.5)[0]
        out = []
        for b in res.boxes:
            x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0].item())
            cls = int(b.cls[0].item())
            if self.cls_keep is None or cls in self.cls_keep:
                out.append((x1,y1,x2-x1,y2-y1,conf,cls))
        # pick best and smooth
        h,w = frame.shape[:2]
        box = max(out, key=lambda b:b[4]) if out else (0,0,w,h,1.0,0)
        # dilate box a bit
        x,y,Bw,Bh,cf,cl = box
        dx,dy = int(0.08*Bw), int(0.08*Bh)
        x = max(0, x-dx); y = max(0, y-dy)
        Bw = min(w-x, Bw+2*dx); Bh = min(h-y, Bh+2*dy)
        cur = np.array([x,y,Bw,Bh], dtype=float)
        if self._prev is None: self._prev = cur
        sm = self._ema*self._prev + (1-self._ema)*cur
        self._prev = sm
        xs,ys,ws,hs = [int(v) for v in sm]
        return [(xs,ys,ws,hs,cf,cl)]
