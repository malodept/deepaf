import argparse, torch, cv2 as cv, numpy as np
from ..models.step_estimator import StepEstimatorTiny
from ..models.focus_discriminator import FocusDiscriminatorTiny
from ..eval.metrics import tenengrad, laplacian_sharpness
from ..roi.detector_tracker import OptionalYOLO

@torch.no_grad()
def predict_absZ(net, img):
    x = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(next(net.parameters()).device)
    return float(torch.relu(net(x)).item())

@torch.no_grad()
def predict_infocus(net, img, thr=0.5):
    x = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(next(net.parameters()).device)
    p = net(x).sigmoid().item()
    return p >= thr, p

# new: disk PSF generator
def _disk_kernel(R: int):
    if R <= 0:
        k = np.zeros((1,1), np.float32); k[0,0] = 1.0
        return k
    d = 2*R + 1
    y, x = np.ogrid[-R:R+1, -R:R+1]
    m = (x*x + y*y) <= R*R
    k = np.zeros((d,d), np.float32); k[m] = 1.0
    k /= k.sum()
    return k

# new: simple defocus + optional breathing (scale) + gamma handling
def _apply_defocus(frame, Z, psf_per_step=0.9, breathing=0.0, gamma=2.2):
    # défocus + breathing simples pour simuler la lentille pendant la démo
    img = frame.astype(np.float32) / 255.0
    img_lin = np.power(img, gamma)
    R = int(max(0, round(abs(Z) * psf_per_step)))
    psf = _disk_kernel(R)
    bl = cv.filter2D(img_lin, -1, psf, borderType=cv.BORDER_REFLECT)
    if abs(breathing) > 1e-6:
        scale = 1.0 + breathing * (1 if Z >= 0 else -1)
        h, w = bl.shape[:2]
        bh, bw = int(round(h*scale)), int(round(w*scale))
        tmp = cv.resize(bl, (bw, bh), interpolation=cv.INTER_LINEAR)
        if scale >= 1.0:
            y0 = (bh - h)//2; x0 = (bw - w)//2
            bl = tmp[y0:y0+h, x0:x0+w]
        else:
            canvas = np.zeros_like(bl)
            y0 = (h - bh)//2; x0 = (w - bw)//2
            canvas[y0:y0+bh, x0:x0+bw] = tmp
            bl = canvas
    out = np.power(bl, 1.0/gamma)
    out = (out * 255.0).clip(0,255).astype(np.uint8)
    return out

def crop(frame, box, size=256):
    x,y,w,h,_,_ = box
    x0,y0,x1,y1 = max(0,x), max(0,y), min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    roi = frame[y0:y1, x0:x1]
    return cv.resize(roi, (size,size), interpolation=cv.INTER_AREA)

def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    step = StepEstimatorTiny().to(dev); step.load_state_dict(torch.load(args.step_w, map_location=dev))
    disc = FocusDiscriminatorTiny().to(dev); disc.load_state_dict(torch.load(args.disc_w, map_location=dev))
    yolo = OptionalYOLO(weights=args.yolo_weights) if args.use_yolo else None

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {args.video}")
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv.CAP_PROP_FPS) or 20
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(args.save_out, fourcc, fps, (w,h))

    Z = args.init_Z
    frame_id, in_focus_frames = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if yolo is not None:
            dets = yolo.detect(frame)
            box = max(dets, key=lambda b:b[4]) if dets else (0,0,w,h,1.0,0)
        else:
            box=(0,0,w,h,1.0,0)

        roi = crop(frame, box, size=args.size)
        is_ok, prob = predict_infocus(disc, roi, thr=args.inf_thr)
        if is_ok:
            status = f"IN-FOCUS p={prob:.2f}"
            step_cmd = 0.0
            in_focus_frames += 1
        else:
            d = predict_absZ(step, roi)
            sgn = -1.0 if Z>0 else 1.0
            Z += sgn * d
            status = f"MOVE {sgn*d:.2f} -> Z={Z:.2f}"
            step_cmd = sgn*d

        g1 = tenengrad(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
        g2 = laplacian_sharpness(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
        x,y,w2,h2,_,_ = box
        cv.rectangle(frame, (x,y), (x+w2,y+h2), (0,255,255), 2)
        cv.putText(frame, status, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.putText(frame, f"TEN={g1:.2f} LAPV={g2:.2f}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        fvis = frame
        if args.simulate_lens:
            fvis = _apply_defocus(frame, Z, args.psf_per_step, args.breathing)
        if args.subject_bokeh:
            # masque doux centré sur la ROI: garde sujet net (Z≈0), floute le fond (Z=bg_Z)
            mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
            x,y,w2,h2,_,_ = box
            cv.rectangle(mask, (x,y), (x+w2,y+h2), 255, -1)
            if args.feather_px>0:
                k = max(3, 2*args.feather_px+1)
                mask = cv.GaussianBlur(mask, (k,k), args.feather_px)
            mask3 = cv.merge([mask,mask,mask]).astype(np.float32)/255.0
            bg_blur = _apply_defocus(frame, args.bg_Z, args.psf_per_step, 0.0)
            fvis = (mask3*fvis + (1-mask3)*bg_blur).astype(np.uint8)
        out.write(fvis)
        frame_id += 1

    cap.release(); out.release()
    ratio = (in_focus_frames / max(1,frame_id))
    print(f"Saved -> {args.save_out} | frames={frame_id} | in-focus ratio={ratio:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--save_out", type=str, default="out_demo.mp4")
    ap.add_argument("--use_yolo", type=int, default=0)
    ap.add_argument("--yolo_weights", type=str, default="yolov8n.pt")
    ap.add_argument("--step_w", type=str, default="runs/step_estimator.pt")
    ap.add_argument("--disc_w", type=str, default="runs/focus_discriminator.pt")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--inf_thr", type=float, default=0.5)
    ap.add_argument("--init_Z", type=float, default=3.0)
    # simulateur global
    ap.add_argument("--simulate_lens", type=int, default=0)
    ap.add_argument("--psf_per_step", type=float, default=0.9)
    ap.add_argument("--breathing", type=float, default=0.0)
    # nouveau: bokeh sujet (fond flou, sujet net)
    ap.add_argument("--subject_bokeh", type=int, default=0)
    ap.add_argument("--feather_px", type=int, default=28)
    ap.add_argument("--bg_Z", type=float, default=3.5)
    args = ap.parse_args()
    main(args)