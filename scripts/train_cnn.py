"""2D CNN on the global-time image of each cadence (CPU).

Each cadence becomes a 2 x 128 x 128 image: panel p, row r is placed at global row
p * (16 + gap) + r, so a real ETI signal is one straight line through the three ON panels.
Channel 0 is the normalized power, channel 1 is +1 on ON rows, -1 on OFF rows, 0 in slew gaps
and dropped rows.

Training augmentation: fresh PKS-style degradation (prob 0.75) and a frequency flip (prob 0.5;
it flips the drift sign, which every class allows). Validation uses the fixed degraded copy
made by the same seeds as scripts/build_features.py. Test uses flip TTA.
    python scripts/train_cnn.py --folds 0 1 2 3 4 --name cnn

--arch swaps the small Net for an ImageNet-pretrained timm backbone (mixed precision, needs
internet on the first run to fetch the public weights):
    python scripts/train_cnn.py --arch efficientnet_b0 --levels --lr 1e-3 --epochs 12 --name eff_lvl
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sig.data import CPUS, LABELS, N_CHAN, N_ROWS, ON, ROOT, SEED, load_folds, load_split  # noqa: E402
from sig.features import panel_levels  # noqa: E402
from sig.prep import degrade, normalize  # noqa: E402

H = 128
CHUNK = 250
P_DEGRADE = 0.75
LEVEL_SCALE = 8.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(CPUS)


def to_image(x: np.ndarray, gaps: np.ndarray, levels: bool = False) -> np.ndarray:
    """uint8 cadences (n, 6, 16, 128) -> float16 images (n, 2 or 3, 128, 128).

    With levels, channel 2 holds each panel's overall brightness (log2 power relative to the
    cadence mean, x8), which the row normalization removes but a scintillating star raises.
    """
    z, valid = normalize(x)
    n = len(z)
    img = np.zeros((n, 3 if levels else 2, H, N_CHAN), np.float16)
    lvl = panel_levels(x)[[f"lvl_p{p}" for p in range(6)]].to_numpy() * LEVEL_SCALE if levels else None
    sign = np.where(ON, 1.0, -1.0)
    for i in range(n):
        for p in range(6):
            r0 = p * (N_ROWS + gaps[i])
            img[i, 0, r0:r0 + N_ROWS] = np.clip(z[i, p], -6, 30)
            img[i, 1, r0:r0 + N_ROWS] = sign[p] * valid[i, p][:, None]
            if levels:
                img[i, 2, r0:r0 + N_ROWS] = lvl[i, p] * valid[i, p][:, None]
    return img


def _job(args: tuple) -> np.ndarray:
    name, start, stop, mode, seed, levels = args
    sp = load_split(name)
    x = np.asarray(sp.x[start:stop])
    if mode == "fixed":
        x = degrade(x, np.random.default_rng([SEED, start]), gain_jitter=levels)
    elif mode == "random":
        rng = np.random.default_rng([seed, start])
        d = degrade(x, rng, gain_jitter=levels)
        pick = rng.random(len(x)) < P_DEGRADE
        x = np.where(pick[:, None, None, None], d, x)
    return to_image(x, sp.gaps[start:stop], levels)


def images(pool: ProcessPoolExecutor, name: str, mode: str, seed: int = 0, levels: bool = False) -> np.ndarray:
    n = len(load_split(name).meta)
    jobs = [(name, a, min(a + CHUNK, n), mode, seed, levels) for a in range(0, n, CHUNK)]
    return np.concatenate(list(pool.map(_job, jobs)))


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    def __init__(self, in_ch: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.b1, self.b2, self.b3, self.b4 = block(32, 48), block(48, 96), block(96, 160), block(160, 256)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(LABELS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.stem(x), 2)          # 64
        x = F.max_pool2d(self.b1(x), 2)            # 32
        x = F.max_pool2d(self.b2(x), 2)            # 16
        x = F.max_pool2d(self.b3(x), 2)            # 8
        x = self.b4(x)
        x = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        return self.head(x)


class Pretrained(nn.Module):
    """A timm backbone with public ImageNet weights (allowed by rule 02); the 128 x 128 image is
    resized to size x size on the GPU. timm adapts the pretrained stem to 2 or 3 input channels."""

    def __init__(self, arch: str, in_ch: int, size: int) -> None:
        super().__init__()
        import timm
        self.size = size
        self.net = timm.create_model(arch, pretrained=True, in_chans=in_ch, num_classes=len(LABELS), drop_rate=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.net(x)


def predict(model: nn.Module, img: np.ndarray, tta: bool = True, amp: bool = False) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(DEVICE.type, enabled=amp):
        for a in range(0, len(img), 256):
            xb = torch.from_numpy(img[a:a + 256].astype(np.float32)).to(DEVICE)
            p = F.softmax(model(xb).float(), 1)
            if tta:
                p = 0.5 * (p + F.softmax(model(torch.flip(xb, [3])).float(), 1))
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--name", default="cnn")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--levels", action="store_true", help="add the panel brightness channel and PKS gain jitter")
    ap.add_argument("--arch", default="small", help="'small' (Net above) or a timm model, e.g. efficientnet_b0")
    ap.add_argument("--size", type=int, default=256, help="input size for a timm model")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit-batches", type=int, default=None, help="smoke test: batches per epoch")
    args = ap.parse_args()
    amp = args.arch != "small" and DEVICE.type == "cuda"

    tr, te = load_split("train"), load_split("test")
    y, folds = tr.y, load_folds()
    w = torch.tensor(len(y) / (len(LABELS) * np.bincount(y, minlength=len(LABELS))), dtype=torch.float32).to(DEVICE)
    pool = ProcessPoolExecutor(CPUS)
    print("device:", DEVICE, " cpus:", CPUS, flush=True)
    (ROOT / "oof").mkdir(exist_ok=True)
    val_img = images(pool, "train", "fixed", levels=args.levels)
    test_img = images(pool, "test", "none", levels=args.levels)
    oof = np.full((len(y), len(LABELS)), np.nan)
    test = np.zeros((len(test_img), len(LABELS)))

    for k in args.folds:
        torch.manual_seed(args.seed + k)
        rng = np.random.default_rng(args.seed + k)
        fit = np.where(folds != k)[0]
        va = np.where(folds == k)[0]
        in_ch = 3 if args.levels else 2
        model = (Net(in_ch) if args.arch == "small" else Pretrained(args.arch, in_ch, args.size)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = args.epochs * (args.limit_batches or int(np.ceil(len(fit) / 128)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        for ep in range(args.epochs):
            t = time.time()
            ep_img = images(pool, "train", "random", seed=args.seed * 1000 + k * 100 + ep, levels=args.levels)
            model.train()
            order = rng.permutation(fit)
            n_b = args.limit_batches or int(np.ceil(len(order) / 128))
            loss_sum = 0.0
            for b in range(n_b):
                idx = order[b * 128:(b + 1) * 128]
                xb = torch.from_numpy(ep_img[idx].astype(np.float32))
                flip = torch.from_numpy(rng.random(len(idx)) < 0.5)
                xb[flip] = torch.flip(xb[flip], [3])
                xb, yb = xb.to(DEVICE), torch.from_numpy(y[idx]).to(DEVICE)
                with torch.autocast(DEVICE.type, enabled=amp):
                    loss = F.cross_entropy(model(xb).float(), yb, weight=w, label_smoothing=0.05)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                loss_sum += loss.item()
            p = predict(model, val_img[va], tta=False, amp=amp)
            print(f"fold {k} ep {ep}: loss {loss_sum / n_b:.4f}  val F1 "
                  f"{f1_score(y[va], p.argmax(1), average='macro'):.4f}  {time.time() - t:.0f}s", flush=True)
        oof[va] = predict(model, val_img[va], amp=amp)
        test += predict(model, test_img, amp=amp) / len(args.folds)
        print(f"fold {k} TTA val F1 {f1_score(y[va], oof[va].argmax(1), average='macro'):.4f}", flush=True)
        torch.save(model.state_dict(), ROOT / f"oof/{args.name}_f{k}.pt")

    done = ~np.isnan(oof[:, 0])
    print(f"OOF macro-F1 (degraded, {done.sum()} rows): {f1_score(y[done], oof[done].argmax(1), average='macro'):.4f}")
    cols = pd.DataFrame(oof, columns=LABELS)
    pd.concat([tr.meta[["id"]], cols], axis=1).to_csv(ROOT / f"oof/{args.name}_oof.csv", index=False)
    pd.concat([te.meta[["id"]], pd.DataFrame(test, columns=LABELS)], axis=1).to_csv(
        ROOT / f"oof/{args.name}_test.csv", index=False)
    pool.shutdown()


if __name__ == "__main__":
    main()
