"""render.py
Render and evaluate a trained SAC checkpoint deterministically for N episodes.

Example
-------
    python render.py \
        --checkpoint logs/.../checkpoint_003354 \
        --render human --episodes 3

Notes
-----
* The script always runs the policy with `explore=False` (deterministic).
* `max_steps` defaults to the environment limit (1500).
* Prints episode‑level returns, success rate, and min–max statistics.
"""

import argparse
import time
import numpy as np
import gymnasium as gym
import ray
from ray.rllib.algorithms.sac import SAC


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Load a Unicycle SAC checkpoint and render deterministic rollouts.")

    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the checkpoint_XXXX directory.")

    parser.add_argument(
        "--render", default="human",
        choices=["human", "rgb_array", "depth_array"],
        help="Gym render_mode (default: human).")

    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run (default: 1).")

    parser.add_argument(
        "--max-steps", type=int, default=1500,
        help="Maximum steps per episode (default: 1500).")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load trained algorithm ------------------------------------------------
    algo = SAC.from_checkpoint(args.checkpoint)

    # 2) Create environment ----------------------------------------------------
    import envs.unicycle  # Registers Unicycle-v0
    env = gym.make("Unicycle-v0", render_mode=args.render)

    returns = []
    successes = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = truncated = False
        total_r, steps = 0.0, 0

        while not (done or truncated) and steps < args.max_steps:
            # deterministic action (no exploration noise)
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            total_r += reward
            steps += 1

            # Real‑time speed adjustment for human render (200 Hz simulator)
            if args.render == "human":
                time.sleep(env.dt)

        returns.append(total_r)
        successes += int(info.get("goal_reached", False))
        print(f"Ep{ep:03d}  steps={steps:<4d}  return={total_r:8.2f}  "
              f"success={info.get('goal_reached', False)}")

    env.close()
    ray.shutdown()

    # -------------------- Summary -------------------------------------------
    returns_np = np.array(returns)
    print("\n=== Evaluation Summary ===")
    print(f"Episodes          : {args.episodes}")
    print(f"Mean return       : {returns_np.mean():.2f} ± {returns_np.std():.2f}")
    print(f"Success rate      : {successes}/{args.episodes} "
          f"({successes/args.episodes*100:.1f}%)")
    print(f"Return (min–max)  : {returns_np.min():.2f} – {returns_np.max():.2f}")


if __name__ == "__main__":
    main()
