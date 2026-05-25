# train/unicycle_train/coach/v5/build_dataset.py

"""
V5 Stage 2: Build Autoencoder Training Dataset

Reads raw per-episode .npz files from dataset/ and produces consolidated
train.npz / val.npz suitable for autoencoder training.

Decisions (see chat history for rationale):
  - Pool A (seeds 2000-2099, 100 students) is split at student level:
      Train: seeds 2000-2079 (80 students)
      Val:   seeds 2080-2099 (20 students)
    Disjoint at student level so autoencoder generalization to unseen
    students can be validated (mirrors human transfer scenario).
  - Episode range per student: 0 to 1999 (2000 episodes).
    All students converge by ep 1412 (max). Cutoff 2000 fully covers
    learning phase + early convergence phase, drops redundant late-
    convergence episodes.
  - Feature per timestep: state (19D) concat action (2D) = 21D.
    Reward excluded to keep latent z reward-independent.
  - Sequence length: fixed 1500 (environment max episode step).
    Zero-padded for shorter episodes; mask marks valid timesteps.
  - Normalization: z-score per feature. Mean/std computed on train set
    only and applied to both splits to avoid data leakage. Stored in
    output npz for inverse-transform / inference reuse.

Output: train.npz, val.npz under dataset_processed/
  features: (N, 1500, 21) float32 — normalized
  masks:    (N, 1500)     bool    — True for valid timesteps
  seeds:    (N,)          int32   — student seed for each episode
  ep_ids:   (N,)          int32   — episode index within student
  feature_mean: (21,)     float32 — for inverse-transform
  feature_std:  (21,)     float32

Usage:
    python -m train.unicycle_train.coach.v5.build_dataset \
        --raw-dir   train/unicycle_train/coach/v5/dataset \
        --output-dir train/unicycle_train/coach/v5/dataset_processed
"""

import argparse
import time
import numpy as np
from pathlib import Path


# Constants matching V5 collection (see recorder.py and train_student.py)
TRAIN_SEEDS = list(range(2000, 2080))  # 80 students
VAL_SEEDS   = list(range(2080, 2100))  # 20 students
EPISODES_PER_STUDENT = 2000            # ep 0 through ep 1999
MAX_SEQ_LEN = 1500                     # environment max episode step
STATE_DIM = 19
ACTION_DIM = 2
FEATURE_DIM = STATE_DIM + ACTION_DIM  # = 21


def load_episode_features(npz_path):
    """Load one episode and return (T, 21) feature array.
    
    Per-timestep feature = concat(state[t], action[t]) for t in 0..T-1.
    states has T+1 entries (initial state plus T transitions), so we drop
    the last state which has no paired action.
    """
    data = np.load(npz_path)
    states = data['states']    # (T+1, 19)
    actions = data['actions']  # (T, 2)
    T = actions.shape[0]
    # Pair state[t] with action[t]
    features = np.concatenate([states[:T], actions], axis=1)  # (T, 21)
    return features.astype(np.float32)


def build_split(raw_dir, seeds, split_name):
    """Load all episodes for the given seeds and pack into padded array.
    
    Returns:
        features: (N, 1500, 21) float32, zero-padded
        masks:    (N, 1500) bool, True for valid timesteps
        ep_seeds: (N,) int32
        ep_ids:   (N,) int32
    """
    n_episodes = len(seeds) * EPISODES_PER_STUDENT
    
    print(f"\n[{split_name}] Allocating arrays for {n_episodes} episodes...")
    print(f"  features: ({n_episodes}, {MAX_SEQ_LEN}, {FEATURE_DIM}) float32")
    mem_gb = n_episodes * MAX_SEQ_LEN * FEATURE_DIM * 4 / 1e9
    print(f"  Approx memory: {mem_gb:.1f} GB")
    
    features = np.zeros((n_episodes, MAX_SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((n_episodes, MAX_SEQ_LEN), dtype=bool)
    ep_seeds = np.zeros(n_episodes, dtype=np.int32)
    ep_ids = np.zeros(n_episodes, dtype=np.int32)
    
    t0 = time.time()
    idx = 0
    skipped = 0
    truncated = 0
    
    for s_i, seed in enumerate(seeds):
        student_dir = Path(raw_dir) / f"student_{seed:04d}"
        if not student_dir.exists():
            print(f"  [WARN] Missing student dir: {student_dir}, skipping")
            skipped += EPISODES_PER_STUDENT
            idx += EPISODES_PER_STUDENT
            continue
        
        for ep_i in range(EPISODES_PER_STUDENT):
            npz_path = student_dir / f"episode_{ep_i:06d}.npz"
            if not npz_path.exists():
                # Leave zeros (mask will be all False, naturally excluded from training)
                skipped += 1
                idx += 1
                continue
            
            feat = load_episode_features(npz_path)  # (T, 21)
            T = feat.shape[0]
            
            if T > MAX_SEQ_LEN:
                # Should not happen given env max_episode_step=1500, but defend
                feat = feat[:MAX_SEQ_LEN]
                T = MAX_SEQ_LEN
                truncated += 1
            
            features[idx, :T, :] = feat
            masks[idx, :T] = True
            ep_seeds[idx] = seed
            ep_ids[idx] = ep_i
            idx += 1
        
        # Progress per student
        elapsed = time.time() - t0
        rate = (s_i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(seeds) - s_i - 1) / rate if rate > 0 else 0
        print(f"  [{s_i+1}/{len(seeds)}] seed {seed}: {elapsed:.0f}s elapsed, "
              f"ETA {eta:.0f}s")
    
    print(f"\n[{split_name}] Loaded {idx - skipped} episodes "
          f"(skipped: {skipped}, truncated: {truncated})")
    print(f"[{split_name}] Total time: {time.time() - t0:.0f}s")
    
    return features, masks, ep_seeds, ep_ids


def compute_normalization(features, masks):
    """Compute per-feature mean and std using only valid (unmasked) timesteps.
    
    Returns:
        mean: (21,) float32
        std:  (21,) float32 (clamped at 1e-6 to avoid div-by-zero)
    """
    print(f"\nComputing normalization from valid timesteps...")
    # Flatten to (total_valid_steps, 21)
    valid = features[masks]  # boolean indexing collapses first two dims
    print(f"  Total valid timesteps: {valid.shape[0]:,}")
    
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    
    print(f"  Mean range: [{mean.min():+.3f}, {mean.max():+.3f}]")
    print(f"  Std range:  [{std.min():.3f}, {std.max():.3f}]")
    
    return mean, std


def normalize_features(features, masks, mean, std):
    """Apply z-score normalization in-place. Padding positions stay 0."""
    # Only normalize valid positions to keep padding as 0
    # (Normalizing padding would turn zeros into -mean/std which corrupts mask semantics)
    features[masks] = (features[masks] - mean) / std
    return features


def main():
    parser = argparse.ArgumentParser(description="V5 Stage 2: Build autoencoder dataset")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to raw dataset/ with student_*/episode_*.npz")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output dir for train.npz and val.npz")
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  V5 Stage 2: Build Autoencoder Training Dataset")
    print("=" * 60)
    print(f"  Raw dir:     {raw_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Train seeds: {TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]} ({len(TRAIN_SEEDS)} students)")
    print(f"  Val seeds:   {VAL_SEEDS[0]}-{VAL_SEEDS[-1]} ({len(VAL_SEEDS)} students)")
    print(f"  Episodes/student: {EPISODES_PER_STUDENT}")
    print(f"  Max seq len: {MAX_SEQ_LEN}")
    print(f"  Feature dim: {FEATURE_DIM} (state {STATE_DIM} + action {ACTION_DIM})")
    print("=" * 60)
    
    # Build train split
    train_feat, train_mask, train_seeds_arr, train_ep_ids = build_split(
        raw_dir, TRAIN_SEEDS, "TRAIN"
    )
    
    # Compute normalization from train only
    mean, std = compute_normalization(train_feat, train_mask)
    
    # Apply normalization to train
    print(f"\nNormalizing train features...")
    train_feat = normalize_features(train_feat, train_mask, mean, std)
    
    # Save train
    train_path = output_dir / "train.npz"
    print(f"\nSaving train split to {train_path}...")
    np.savez(
        train_path,
        features=train_feat,
        masks=train_mask,
        seeds=train_seeds_arr,
        ep_ids=train_ep_ids,
        feature_mean=mean,
        feature_std=std,
    )
    print(f"  Saved. Size on disk: {train_path.stat().st_size / 1e9:.2f} GB")
    
    # Free memory before building val
    del train_feat, train_mask, train_seeds_arr, train_ep_ids
    
    # Build val split
    val_feat, val_mask, val_seeds_arr, val_ep_ids = build_split(
        raw_dir, VAL_SEEDS, "VAL"
    )
    
    # Apply train normalization to val (no recomputation)
    print(f"\nNormalizing val features with train statistics...")
    val_feat = normalize_features(val_feat, val_mask, mean, std)
    
    # Save val
    val_path = output_dir / "val.npz"
    print(f"\nSaving val split to {val_path}...")
    np.savez(
        val_path,
        features=val_feat,
        masks=val_mask,
        seeds=val_seeds_arr,
        ep_ids=val_ep_ids,
        feature_mean=mean,
        feature_std=std,
    )
    print(f"  Saved. Size on disk: {val_path.stat().st_size / 1e9:.2f} GB")
    
    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()