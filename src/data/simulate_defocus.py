
import os, argparse, random
import numpy as np
import cv2 as cv
from pathlib import Path
from tqdm import tqdm

def disk_kernel(radius: int):
    # Create a normalized disk PSF kernel
    if radius <= 0:
        k = np.zeros((1,1), np.float32); k[0,0]=1.0
        return k
    d = 2*radius+1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y <= radius*radius
    k = np.zeros((d,d), np.float32)
    k[mask] = 1.0
    s = k.sum()
    if s > 0: k /= s
    return k

def apply_defocus(img, r_pix: float, breathing: float=0.0, gamma: float=2.2):
    # Approximate optical defocus + breathing + gamma pipeline.
    h, w = img.shape[:2]
    img_lin = np.power(np.clip(img/255.0, 0, 1), gamma).astype(np.float32)
    R = int(max(0, round(r_pix)))
    psf = disk_kernel(R)
    blurred = cv.filter2D(img_lin, -1, psf, borderType=cv.BORDER_REFLECT)
    if abs(breathing) > 1e-6:
        scale = 1.0 + breathing
        bh, bw = int(round(h*scale)), int(round(w*scale))
        tmp = cv.resize(blurred, (bw, bh), interpolation=cv.INTER_LINEAR)
        if scale >= 1.0:
            y0 = (bh - h)//2; x0 = (bw - w)//2
            tmp = tmp[y0:y0+h, x0:x0+w]
        else:
            canvas = np.zeros_like(blurred)
            y0 = (h - bh)//2; x0 = (w - bw)//2
            canvas[y0:y0+bh, x0:x0+bw] = tmp
            tmp = canvas
        blurred = tmp
    noise = np.random.normal(0, 0.002, size=blurred.shape).astype(np.float32)
    blurred = np.clip(blurred + noise, 0, 1)
    out = np.power(blurred, 1.0/gamma)
    out = np.clip(out*255.0, 0, 255).astype(np.uint8)
    return out

def load_images_random(n: int, size=512):
    imgs = []
    for i in range(n):
        canvas = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        for _ in range(8):
            color = tuple(int(x) for x in np.random.randint(0,255,3))
            if random.random()<0.5:
                p1 = (random.randint(0,size-1), random.randint(0,size-1))
                p2 = (random.randint(0,size-1), random.randint(0,size-1))
                cv.rectangle(canvas, p1, p2, color, thickness=random.randint(1,4))
            else:
                c = (random.randint(0,size-1), random.randint(0,size-1))
                r = random.randint(5, size//4)
                cv.circle(canvas, c, r, color, thickness=random.randint(1,4))
        imgs.append(canvas)
    return imgs

def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = []
    imgs = load_images_random(args.n, size=args.size)
    for i, img in enumerate(tqdm(imgs, desc="Synth")):
        absZ = random.uniform(0, args.max_absZ)
        r_pix = absZ * args.psf_per_step
        breathing = random.uniform(-args.max_breathing, args.max_breathing)
        lab_step = absZ
        lab_disc = 1 if absZ < args.focus_thr else 0
        img_def = apply_defocus(img, r_pix=r_pix, breathing=breathing, gamma=args.gamma)
        base = f"synth_{i:05d}"
        cv.imwrite(str(out/f"{base}_sharp.png"), img)
        cv.imwrite(str(out/f"{base}_def.png"), img_def)
        meta.append({
            "base": base,
            "absZ": float(lab_step),
            "in_focus": int(lab_disc),
            "r_pix": float(r_pix),
            "breathing": float(breathing)
        })
    with open(out/"meta.json", "w") as f:
        import json; json.dump(meta, f, indent=2)
    print(f"Wrote {len(meta)} samples to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data_synth")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--max_absZ", type=float, default=8.0)
    p.add_argument("--psf_per_step", type=float, default=0.9)
    p.add_argument("--max_breathing", type=float, default=0.03)
    p.add_argument("--focus_thr", type=float, default=0.6)
    p.add_argument("--gamma", type=float, default=2.2)
    args = p.parse_args()
    main(args)
