"""render.py
Render and evaluate a trained SAC checkpoint deterministically for N episodes.

Example
-------
    python render.py \
        --checkpoint logs/.../checkpoint_003354 \
        --render human --episodes 3
"""
import argparse
import time
import numpy as np
import gymnasium as gym
import ray
from ray import tune
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
        "--episodes", type=int, default=100,
        help="Number of episodes to run (default: 100).")

    parser.add_argument(
        "--max-steps", type=int, default=1500,
        help="Maximum steps per episode (default: 1500).")

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Register custom env so RLlib can recreate it inside workers
    # ------------------------------------------------------------------ #
    def unicycle_env_creator(cfg):
        import gymnasium as gym
        import envs.unicycle          # registers Unicycle-v0
        return gym.make("Unicycle-v0", **cfg)

    tune.register_env("UnicycleRLlib", unicycle_env_creator)

    # 1) Load trained algorithm -----------------------------------------
    algo = SAC.from_checkpoint(args.checkpoint)

    # 2) Create local env for rendering ---------------------------------
    env = gym.make("Unicycle-v0", render_mode=args.render)

    returns, successes = [], 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = truncated = False
        total_r, steps = 0.0, 0

        while not (done or truncated) and steps < args.max_steps:
            action = algo.compute_single_action(obs, explore=False)  # deterministic
            obs, reward, done, truncated, info = env.step(action)
            total_r += reward
            steps += 1

            if args.render == "human":          # real-time speed for GUI
                time.sleep(env.dt)

        returns.append(total_r)
        successes += int(info.get("goal_reached", False))
        print(f"Ep{ep:03d}  steps={steps:<4d}  "
              f"return={total_r:8.2f}  success={info.get('goal_reached', False)}")

    env.close()
    ray.shutdown()

    # -------------------- Summary --------------------------------------
    r = np.array(returns)
    print("\n=== Evaluation Summary ===")
    print(f"Episodes          : {args.episodes}")
    print(f"Mean return       : {r.mean():.2f} ± {r.std():.2f}")
    print(f"Success rate      : {successes}/{args.episodes} "
          f"({successes/args.episodes*100:.1f}%)")
    print(f"Return (min–max)  : {r.min():.2f} – {r.max():.2f}")


if __name__ == "__main__":
    main()
