"""
Auto-focus / approach detection from a video (YOLOv8 + DeepSORT + MiDaS + Kalman)

Usage:
    pip install ultralytics deep-sort-realtime torch torchvision timm opencv-python filterpy

    python autofocus_hybrid.py --input input.mp4 --output out.mp4 --use_depth

Notes:
 - The script tries to load YOLOv8 via ultralytics and MiDaS via torch.hub.
 - If MiDaS (depth) can't be loaded, the code falls back to using bounding-box area changes.
 - For tracking, it uses deep-sort-realtime. If deep-sort is not installed, there is a simple centroid tracker fallback.
 - Depth estimation is relatively slow on CPU; a GPU will speed it up.

"""

import argparse
import time
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# Kalman helper (using OpenCV Kalman)
class SimpleKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)  # [x,y,w,h,vx,vy] -> measure x,y,w
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0],
            [0,1,0,0,0,1],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.zeros((3,6), dtype=np.float32)
        self.kf.measurementMatrix[0,0] = 1
        self.kf.measurementMatrix[1,1] = 1
        self.kf.measurementMatrix[2,2] = 1
        cv2.setIdentity(self.kf.processNoiseCov, 1e-2)
        cv2.setIdentity(self.kf.measurementNoiseCov, 1e-1)
        cv2.setIdentity(self.kf.errorCovPost, 1.)
        self.initialized = False

    def predict(self):
        return self.kf.predict()

    def correct(self, x, y, w):
        z = np.array([[np.float32(x)], [np.float32(y)], [np.float32(w)]])
        if not self.initialized:
            self.kf.statePost[:3,0] = np.array([x,y,w], dtype=np.float32)
            self.initialized = True
        return self.kf.correct(z)

# Depth model loader: MiDaS ------------------------------------------------------
class DepthEstimator:
    def __init__(self, device='gpu'):
        self.device = device
        self.model = None
        self.transform = None
        if torch is None:
            print('Torch not available: depth disabled')
            return
        try:
            midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
            midas.to(device).eval()
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            transform = midas_transforms.default_transform
            self.model = midas
            self.transform = transform
            print('MiDaS loaded')
        except Exception as e:
            print('Failed to load MiDaS:', e)
            self.model = None

    def estimate(self, frame, bbox=None):
        # frame: BGR uint8
        if self.model is None:
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img.shape[:2], mode='bicubic', align_corners=False).squeeze()
            depth_map = prediction.cpu().numpy()
        if bbox is not None:
            x1,y1,x2,y2 = bbox
            x1, y1 = max(0,int(x1)), max(0,int(y1))
            x2, y2 = min(frame.shape[1]-1,int(x2)), min(frame.shape[0]-1,int(y2))
            crop = depth_map[y1:y2+1, x1:x2+1]
            if crop.size == 0:
                return None
            return float(np.median(crop))
        return depth_map

# Main processing ----------------------------------------------------------------

def process_video(input_path, output_path=None, use_depth=True, device='cpu'):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    yolo = None
    if YOLO is not None:
        try:
            yolo = YOLO('yolov8n.pt')  # small model
        except Exception as e:
            print('Failed to init YOLO:', e)
            yolo = None

    tracker = None
    if DeepSort is not None:
        tracker = DeepSort(max_age=30)
    else:
        tracker = SimpleCentroidTracker(max_lost=30)

    depth_estimator = DepthEstimator(device=device) if use_depth and torch is not None else None

    # per-track history
    track_hist = {}  # track_id -> deque of dicts (time, bbox, area, depth)
    track_kalman = {}  # track_id -> SimpleKalman

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t0 = time.time()

        # run detector
        detections = []  # list of [x1,y1,x2,y2,score,class]
        if yolo is not None:
            results = yolo(frame, imgsz=640, device=device)
            # results may be a list
            r = results[0]
            boxes = r.boxes
            for b in boxes:
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                score = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                detections.append([x1,y1,x2,y2,score,cls])
        else:
            # fallback: no detections
            detections = []

        # tracking
        tracked_objects = []  # list of dicts {id,bbox,score}
        if DeepSort is not None and len(detections) > 0:
            ds_inputs = []
            for d in detections:
                x1,y1,x2,y2,score,cls = d
                ds_inputs.append(((x1,y1,x2-x1,y2-y1), score, cls))
            tracks = tracker.update_tracks(ds_inputs, frame=frame)
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                tid = tr.track_id
                l,t,w_box,h_box = tr.to_ltwh()
                tracked_objects.append({'id': tid, 'bbox': [l,t,l+w_box,t+h_box], 'score': tr.det_conf})
        else:
            # fallback to simple centroid tracker: map detections to tracker
            assigned = tracker.update(detections)
            # assigned is list of (id, centroid)
            for aid, centroid in assigned:
                # find detection whose centroid matches
                for d in detections:
                    x1,y1,x2,y2,score,cls = d
                    cx,cy = (x1+x2)/2, (y1+y2)/2
                    if abs(cx-centroid[0])<20 and abs(cy-centroid[1])<20:
                        tracked_objects.append({'id': aid, 'bbox':[x1,y1,x2,y2], 'score': score})
                        break

        # for each tracked object, estimate depth or area and update history
        for obj in tracked_objects:
            tid = obj['id']
            x1,y1,x2,y2 = obj['bbox']
            w_box = max(1, x2-x1); h_box = max(1, y2-y1)
            area = w_box*h_box
            depth = None
            if depth_estimator is not None and depth_estimator.model is not None:
                try:
                    depth = depth_estimator.estimate(frame, bbox=(x1,y1,x2,y2))
                except Exception as e:
                    depth = None
            # store history
            if tid not in track_hist:
                track_hist[tid] = deque(maxlen=10)
            track_hist[tid].append({'t': frame_idx/fps, 'bbox':[x1,y1,x2,y2], 'area':area, 'depth':depth})

            # Kalman per track
            if tid not in track_kalman:
                track_kalman[tid] = SimpleKalman()
            kf = track_kalman[tid]
            kf.correct((x1+x2)/2.0, (y1+y2)/2.0, w_box)
            pred = kf.predict()

            # decide approach/away
            decision = 'unknown'
            hist = track_hist[tid]
            if len(hist) >= 2:
                last = hist[-1]
                prev = hist[-2]
                if last['depth'] is not None and prev['depth'] is not None:
                    dz = last['depth'] - prev['depth']
                    # for MiDaS, smaller value = closer (relative scale), check sign
                    # heuristic: if median depth decreased -> approaching
                    if dz < -0.01:
                        decision = 'approaching'
                    elif dz > 0.01:
                        decision = 'moving away'
                    else:
                        decision = 'stable'
                else:
                    # fallback to area change
                    ratio = last['area']/prev['area'] if prev['area']>0 else 1.0
                    if ratio > 1.05:
                        decision = 'approaching'
                    elif ratio < 0.95:
                        decision = 'moving away'
                    else:
                        decision = 'stable'

            # draw
            x1i,y1i,x2i,y2i = map(int,[x1,y1,x2,y2])
            cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), (0,255,0), 2)
            cv2.putText(frame, f'ID {tid} {decision}', (x1i, max(0,y1i-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # draw predicted center
            cx = int(pred[0]); cy = int(pred[1])
            cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)

        # write frame
        if writer is not None:
            writer.write(frame)

        # show
        cv2.imshow('autofocus', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='out.mp4')
    p.add_argument('--no-depth', dest='use_depth', action='store_false')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    process_video(args.input, args.output, use_depth=args.use_depth, device=args.device)
