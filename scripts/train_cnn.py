"""2D CNN on the global-time image of each cadence (CPU).

Each cadence becomes a 2 x 128 x 128 image: panel p, row r is placed at global row
p * (16 + gap) + r, so a real ETI signal is one straight line through the three ON panels.
Channel 0 is the normalized power, channel 1 is +1 on ON rows, -1 on OFF rows, 0 in slew gaps
and dropped rows.

Training augmentation: fresh PKS-style degradation (prob 0.75) and a frequency flip (prob 0.5;
it flips the drift sign, which every class allows). Validation uses the fixed degraded copy
made by the same seeds as scripts/build_features.py. Test uses flip TTA.
    python scripts/train_cnn.py --folds 0 1 2 3 4 --name cnn
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
from sig.prep import degrade, normalize  # noqa: E402

H = 128
CHUNK = 250
P_DEGRADE = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(CPUS)


def to_image(x: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """uint8 cadences (n, 6, 16, 128) -> float16 images (n, 2, 128, 128)."""
    z, valid = normalize(x)
    n = len(z)
    img = np.zeros((n, 2, H, N_CHAN), np.float16)
    sign = np.where(ON, 1.0, -1.0)
    for i in range(n):
        for p in range(6):
            r0 = p * (N_ROWS + gaps[i])
            img[i, 0, r0:r0 + N_ROWS] = np.clip(z[i, p], -6, 30)
            img[i, 1, r0:r0 + N_ROWS] = sign[p] * valid[i, p][:, None]
    return img


def _job(args: tuple) -> np.ndarray:
    name, start, stop, mode, seed = args
    sp = load_split(name)
    x = np.asarray(sp.x[start:stop])
    if mode == "fixed":
        x = degrade(x, np.random.default_rng([SEED, start]))
    elif mode == "random":
        rng = np.random.default_rng([seed, start])
        d = degrade(x, rng)
        pick = rng.random(len(x)) < P_DEGRADE
        x = np.where(pick[:, None, None, None], d, x)
    return to_image(x, sp.gaps[start:stop])


def images(pool: ProcessPoolExecutor, name: str, mode: str, seed: int = 0) -> np.ndarray:
    n = len(load_split(name).meta)
    jobs = [(name, a, min(a + CHUNK, n), mode, seed) for a in range(0, n, CHUNK)]
    return np.concatenate(list(pool.map(_job, jobs)))


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
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


def predict(model: nn.Module, img: np.ndarray, tta: bool = True) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
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
    ap.add_argument("--limit-batches", type=int, default=None, help="smoke test: batches per epoch")
    args = ap.parse_args()

    tr, te = load_split("train"), load_split("test")
    y, folds = tr.y, load_folds()
    w = torch.tensor(len(y) / (len(LABELS) * np.bincount(y, minlength=len(LABELS))), dtype=torch.float32).to(DEVICE)
    pool = ProcessPoolExecutor(CPUS)
    print("device:", DEVICE, " cpus:", CPUS, flush=True)
    (ROOT / "oof").mkdir(exist_ok=True)
    val_img = images(pool, "train", "fixed")
    test_img = images(pool, "test", "none")
    oof = np.full((len(y), len(LABELS)), np.nan)
    test = np.zeros((len(test_img), len(LABELS)))

    for k in args.folds:
        torch.manual_seed(args.seed + k)
        rng = np.random.default_rng(args.seed + k)
        fit = np.where(folds != k)[0]
        va = np.where(folds == k)[0]
        model = Net().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = args.epochs * (args.limit_batches or int(np.ceil(len(fit) / 128)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.15)
        for ep in range(args.epochs):
            t = time.time()
            ep_img = images(pool, "train", "random", seed=args.seed * 1000 + k * 100 + ep)
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
                loss = F.cross_entropy(model(xb), yb, weight=w, label_smoothing=0.05)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.item()
            p = predict(model, val_img[va], tta=False)
            print(f"fold {k} ep {ep}: loss {loss_sum / n_b:.4f}  val F1 "
                  f"{f1_score(y[va], p.argmax(1), average='macro'):.4f}  {time.time() - t:.0f}s", flush=True)
        oof[va] = predict(model, val_img[va])
        test += predict(model, test_img) / len(args.folds)
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
