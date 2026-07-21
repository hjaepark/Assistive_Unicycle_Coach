# train/student/sort_student.py

import re
from pathlib import Path
import numpy as np

DATA_DIR = Path("train/student/data/student")
START_EP = 400

pool = {}
for run_dir in sorted(DATA_DIR.glob("seed*")):
    seed = int(re.search(r"seed(\d+)", run_dir.name).group(1))
    npz = run_dir / f"eval_ep{START_EP:04d}.npz"
    if npz.exists():
        pool[seed] = float(np.load(npz)["successes"].mean())

seeds = np.array(sorted(pool))
succ  = np.array([pool[s] for s in seeds])

good = succ >= 0.90
mid  = (succ >= 0.10) & (succ < 0.90)
poor = succ < 0.10

print(f"Pool size: {len(seeds)} seeds, all resumed from episode {START_EP}\n")
print(f"{'tier':>10} {'count':>6} {'pct':>5}")
print(f"{'good':>10} {good.sum():>6} {good.mean()*100:>4.0f}%")
print(f"{'mid':>10} {mid.sum():>6} {mid.mean()*100:>4.0f}%")
print(f"{'poor':>10} {poor.sum():>6} {poor.mean()*100:>4.0f}%")

tier_of = np.where(good, "good", np.where(mid, "mid", "poor"))
out = DATA_DIR.parent / "pool_ep400.csv"
with open(out, "w") as f:
    f.write("seed,start_ep,solo_success,tier\n")
    for s, sr, t in zip(seeds, succ, tier_of):
        f.write(f"{s},{START_EP},{sr:.4f},{t}\n")
print(f"\n[SAVE] {out}")