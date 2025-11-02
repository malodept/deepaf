
import argparse, torch
import numpy as np, cv2 as cv
from pathlib import Path
from ..models.step_estimator import StepEstimatorTiny
from ..models.focus_discriminator import FocusDiscriminatorTiny
from ..eval.metrics import tenengrad, laplacian_sharpness
from ..roi.detector_tracker import OptionalYOLO

@torch.no_grad()
def predict_absZ(net, img):
    x = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    x = x.to(next(net.parameters()).device)
    p = net(x)
    return float(torch.relu(p).item())

@torch.no_grad()
def predict_infocus(net, img, thr=0.5):
    x = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    x = x.to(next(net.parameters()).device)
    p = net(x).sigmoid().item()
    return p >= thr, p

def crop(frame, box, size=256):
    x,y,w,h,_,_ = box
    x0,y0,x1,y1 = max(0,x), max(0,y), min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    roi = frame[y0:y1, x0:x1].copy()
    roi = cv.resize(roi, (size,size), interpolation=cv.INTER_AREA)
    return roi

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = StepEstimatorTiny().to(device); step.load_state_dict(torch.load(args.step_w, map_location=device))
    disc = FocusDiscriminatorTiny().to(device); disc.load_state_dict(torch.load(args.disc_w, map_location=device))

    cap = cv.VideoCapture(args.video) if args.video else None
    yolo = OptionalYOLO(weights=args.yolo_weights) if args.use_yolo else None

    Z = args.init_Z

    def loop_on_frame(frame):
        nonlocal Z
        if yolo is not None:
            dets = yolo.detect(frame)
            box = max(dets, key=lambda b:b[4]) if dets else (0,0,frame.shape[1],frame.shape[0],1.0,0)
        else:
            h,w = frame.shape[:2]; box=(0,0,w,h,1.0,0)

        roi = crop(frame, box, size=args.size)
        is_ok, prob = predict_infocus(disc, roi, thr=args.inf_thr)
        if is_ok:
            status = f"IN-FOCUS p={prob:.2f}"
            step_cmd = 0.0
        else:
            d = predict_absZ(step, roi)
            sgn = -1.0 if Z>0 else 1.0
            Z += sgn * d
            status = f"MOVE {sgn*d:.2f} -> Z={Z:.2f}"
            step_cmd = sgn*d

        g1 = tenengrad(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
        g2 = laplacian_sharpness(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))

        cv.putText(frame, status, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.putText(frame, f"TEN={g1:.2f} LAPV={g2:.2f}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        x,y,w,h,_,_ = box
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        return frame, step_cmd

    if cap is None:
        print("No video provided. Use --video path.mp4 for a demo.")
        return

    while True:
        ok, frame = cap.read()
        if not ok: break
        vis, _ = loop_on_frame(frame)
        cv.imshow("DeepAF demo", vis)
        if cv.waitKey(1)&0xFF==27: break
    cap.release(); cv.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default="")
    p.add_argument("--use_yolo", type=int, default=0)
    p.add_argument("--yolo_weights", type=str, default="yolov8n.pt")
    p.add_argument("--step_w", type=str, default="runs/step_estimator.pt")
    p.add_argument("--disc_w", type=str, default="runs/focus_discriminator.pt")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--inf_thr", type=float, default=0.5)
    p.add_argument("--init_Z", type=float, default=3.0)
    args = p.parse_args()
    main(args)
