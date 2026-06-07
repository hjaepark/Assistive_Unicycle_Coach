"""Identify the most-hurt seeds and their training-time intervention rate."""
import json, glob
import numpy as np

# Collect coached / uncoached eval success per seed
coached, uncoached = {}, {}
for f in glob.glob("train/unicycle_train/coach/v5/eval_logs/*/with_coach/seed_*/result.json"):
    d = json.load(open(f))
    coached[d["seed"]] = d
for f in glob.glob("train/unicycle_train/coach/v5/eval_logs/*/no_coach/seed_*/result.json"):
    d = json.load(open(f))
    uncoached[d["seed"]] = d["evaluation"]["success_rate"]

rows = []
for seed, d in coached.items():
    if seed not in uncoached:
        continue
    wc = d["evaluation"]["success_rate"]
    nc = uncoached[seed]
    delta = wc - nc

    # Training-time intervention rate: fraction of coaching-phase episodes with w_t==1
    logs = d["training"]["episode_logs"]
    coaching = [e for e in logs if e.get("phase") == "coaching"]
    interv = np.mean([e["w_t"] for e in coaching]) if coaching else float("nan")

    rows.append((seed, nc, wc, delta, interv))

# Sort by delta (most hurt first)
rows.sort(key=lambda r: r[3])

print(f"{'seed':>5} {'uncoach':>8} {'coach':>7} {'delta':>7} {'interv%':>8}")
print("-" * 40)
for seed, nc, wc, delta, interv in rows[:15]:
    print(f"{seed:>5} {nc*100:>7.0f}% {wc*100:>6.0f}% {delta*100:>+6.0f}pp {interv*100:>7.0f}%")
