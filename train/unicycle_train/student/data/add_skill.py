import pandas as pd
import numpy as np
from pathlib import Path

IN_CSV  = Path("rl_student_trajectory_metrics.csv")
OUT_CSV = Path("rl_student_trajectory_metrics_with_skill.csv")

D_GOAL = 5.0
ALPHA = 3.0
DELTA_SOFT_C = 0.2

W_D, W_F, W_E = 0.5, 0.3, 0.2  # Sum to 1

EFFORT_P_LO = 0.05
EFFORT_P_HI = 0.95
EPS = 1e-9

def log_norm_distance(d: np.ndarray, d_goal: float, alpha: float) -> np.ndarray:
    """D' = log(1+alpha*D) / log(1+alpha*D_goal), clipped to [0,1]."""
    denom = np.log1p(alpha * d_goal)
    if denom <= 0:
        return np.clip(d / max(d_goal, EPS), 0, 1)
    return np.clip(np.log1p(alpha * d) / denom, 0, 1)

def robust_minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """x' = clip((x-lo)/(hi-lo), 0, 1)"""
    return np.clip((x - lo) / (hi - lo + EPS), 0, 1)

def soft_gate(dn: np.ndarray, c: float) -> np.ndarray:
    """g(D') = D'/(D'+c)"""
    return dn / (dn + c + EPS)

def main():
    df = pd.read_csv(IN_CSV)

    e_lo = float(df["control_effort"].quantile(EFFORT_P_LO))
    e_hi = float(df["control_effort"].quantile(EFFORT_P_HI))

    dn = log_norm_distance(df["maximum_distance"].to_numpy(dtype=float), D_GOAL, ALPHA)
    fn = np.clip(df["action_flip_rate"].to_numpy(dtype=float), 0, 1)  # 이미 0~1로 가정
    en = robust_minmax(df["control_effort"].to_numpy(dtype=float), e_lo, e_hi)

    df["distance_norm"] = dn
    df["effort_norm"] = en
    df["flip_norm"] = fn

    df["skill_v0"] = (W_D * dn) + (W_F * (1 - fn)) + (W_E * (1 - en))
    df["skill_v1"] = (W_D * dn) + dn * ((W_F * (1 - fn)) + (W_E * (1 - en)))

    g = soft_gate(dn, DELTA_SOFT_C)
    df["skill_v2"] = (W_D * dn) + g * ((W_F * (1 - fn)) + (W_E * (1 - en)))

    df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV.resolve())
    print("\nSkill summary:")
    print(df[["skill_v0","skill_v1","skill_v2"]].describe())

    print("\nMean skill by termination:")
    print(df.groupby("termination")[["skill_v0","skill_v1","skill_v2"]].mean().sort_values("skill_v2", ascending=False))

    low_dn = df.nsmallest(20, "distance_norm")[["run_id","episode","termination","maximum_distance","distance_norm","skill_v0","skill_v1","skill_v2"]]
    print("\nLowest 20 distance_norm episodes:")
    print(low_dn.to_string(index=False))

    print("\nNormalization constants used:")
    print({
        "D_GOAL": D_GOAL, "ALPHA": ALPHA,
        "effort_lo(q05)": e_lo, "effort_hi(q95)": e_hi,
        "weights": (W_D, W_F, W_E),
        "soft_c": DELTA_SOFT_C
    })

if __name__ == "__main__":
    main()
