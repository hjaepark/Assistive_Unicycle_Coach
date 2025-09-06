# train/unicycle_train/evaluate.py

from typing import Callable, Optional, Dict, Any
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

def eval_once(
    model,
    make_env_fn: Optional[Callable[[], gym.Env]] = None,
    env_id: Optional[str] = None,
    n_episodes: int = 100,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, float]:
    
    if make_env_fn is None:
        if env_id is None:
            env_id = "UnicyclePendulumPD-v0"
        def make_env():
            return gym.make(env_id)
        make_env_fn = make_env

    env = make_env_fn()
    rewards, success, fall = [], [], []
    final_xs, final_ys, final_yaws = [], [], []

    for ep in range(n_episodes):
        try:
            obs, info = env.reset(seed=None if seed is None else seed + ep)
        except TypeError:
            obs, info = env.reset()
        ep_rew = 0.0
        last_info: Dict[str, Any] = {}

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            ep_rew += float(r)
            last_info = info
            if terminated or truncated:
                break

        rewards.append(ep_rew)
        success.append(1.0 if bool(last_info.get("goal_reached", False)) else 0.0)
        fall.append(1.0 if bool(last_info.get("is_fallen", False)) else 0.0)
        final_xs.append(float(last_info.get("x_position", 0.0)))
        final_ys.append(float(last_info.get("y_position", 0.0)))
        final_yaws.append(float(last_info.get("unicycle_yaw", 0.0)))

    env.close()

    mean = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return {
        "episode_reward_mean": mean(rewards),
        "success_mean": mean(success),
        "fall_mean": mean(fall),
        "final_x_position_mean": mean(final_xs),
        "final_y_position_mean": mean(final_ys),
        "final_yaw_mean": mean(final_yaws),
    }


class EvalCallBack(BaseCallback):
    def __init__(
        self,
        make_env_fn: Optional[Callable[[], gym.Env]] = None,
        env_id: Optional[str] = None,
        eval_interval_steps: int = 50_000,
        n_eval_episodes: int = 100,
        target_success_rate: float = 0.95,
        best_model_path: Optional[str] = None,
        deterministic: bool = True,
        seed: Optional[int] = None,
        verbose: int = 1,
        require_consecutive: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.make_env_fn = make_env_fn
        self.env_id = env_id
        self.eval_interval_steps = int(eval_interval_steps)
        self.n_eval_episodes = int(n_eval_episodes)
        self.target_success_rate = float(target_success_rate)
        self.best_model_path = str(best_model_path) if best_model_path is not None else None
        self.deterministic = bool(deterministic)
        self.seed = seed
        self.require_consecutive = int(require_consecutive)
        self._success_streak = 0

        self._last_eval_step = 0
        self._best_success = -1.0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) < self.eval_interval_steps:
            return True

        m = eval_once(
            model=self.model,
            make_env_fn=self.make_env_fn,
            env_id=self.env_id,
            n_episodes=self.n_eval_episodes,
            seed=self.seed,
            deterministic=self.deterministic,
        )

        if self.verbose:
            print(
                f"[Eval] steps={self.num_timesteps:,} | eps={self.n_eval_episodes} | "
                f"reward={m['episode_reward_mean']:.2f} | x_final={m['final_x_position_mean']:.2f}m | "
                f"success={m['success_mean']*100:.1f}% | fall={m['fall_mean']*100:.1f}%"
            )

        self.logger.record("eval/episode_reward_mean", m["episode_reward_mean"])
        self.logger.record("eval/success_mean", m["success_mean"])
        self.logger.record("eval/fall_mean", m["fall_mean"])
        self.logger.record("eval/final_x_position_mean", m["final_x_position_mean"])
        self.logger.record("eval/final_y_position_mean", m["final_y_position_mean"])
        self.logger.record("eval/final_yaw_mean", m["final_yaw_mean"])
        self.logger.dump(self.num_timesteps)

        if self.best_model_path and m["success_mean"] > self._best_success:
            self.model.save(self.best_model_path)
            self._best_success = m["success_mean"]
            if self.verbose:
                print(f"[Eval] New best model saved to: {self.best_model_path} "
                      f"(success={self._best_success*100:.1f}%)")

        self._last_eval_step = self.num_timesteps

        if m["success_mean"] >= self.target_success_rate:
            self._success_streak += 1
        else:
            self._success_streak = 0

        if self._success_streak >= self.require_consecutive:
            if self.verbose:
                print("\n" + "=" * 80)
                print("[Training Complete] Target success rate reached "
                    f"{self._success_streak} consecutive time(s)")
                print(f"Final Success Rate: {m['success_mean']*100:.1f}% "
                    f"(Target: {self.target_success_rate*100:.0f}%)")
                print(f"Timesteps: {self.num_timesteps:,}")
                print("=" * 80 + "\n")
            return False

        return True