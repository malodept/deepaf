
import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from .step_estimator import StepEstimatorTiny
from .dataset_util import SynthDefocusDataset

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_tr = SynthDefocusDataset(args.data, split="train", img_size=args.size, task="reg")
    ds_va = SynthDefocusDataset(args.data, split="val", img_size=args.size, task="reg")
    tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    net = StepEstimatorTiny().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()
    best = 1e9; out = Path("runs"); out.mkdir(exist_ok=True)

    for ep in range(1, args.epochs+1):
        net.train(); loss_tr = 0.0
        for x,y in tqdm(tr, desc=f"train ep{ep}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); p = net(x); loss = loss_fn(p,y)
            loss.backward(); opt.step()
            loss_tr += loss.item()*x.size(0)
        loss_tr/=len(ds_tr)

        net.eval(); loss_va = 0.0
        with torch.no_grad():
            for x,y in va:
                x,y = x.to(device), y.to(device)
                p = net(x); loss_va += loss_fn(p,y).item()*x.size(0)
        loss_va/=len(ds_va)
        print(f"ep {ep}: train {loss_tr:.4f} val {loss_va:.4f}")
        if loss_va < best:
            best = loss_va
            torch.save(net.state_dict(), out/"step_estimator.pt")
    print("saved -> runs/step_estimator.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()
    main(args)
