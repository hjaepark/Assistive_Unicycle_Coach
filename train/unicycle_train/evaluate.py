from typing import Any, Dict, Optional

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.env_runner import EnvRunner
from ray.tune.stopper import Stopper
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
)


def _last_info(episode) -> Dict[str, Any]:
    try:
        infos = episode.get_infos(-1)
        if isinstance(infos, dict):
            return infos
        if isinstance(infos, (list, tuple)):
            for i in range(len(infos) - 1, -1, -1):
                if isinstance(infos[i], dict):
                    return infos[i]
    except Exception:
        pass
    return {}


def _extract_mean(result_block: Dict[str, Any], key: str, *, fallback: Dict[str, Any] | None = None) -> float:
    if key in result_block:
        v = result_block[key]
        if isinstance(v, dict):
            return float(v.get("mean", 0.0))
        return float(v)
    if fallback:
        return float(fallback.get(f"{key}_mean", 0.0))
    return 0.0


class SuccessCallback(RLlibCallback):
    def on_episode_end(
        self,
        *,
        env_runner: EnvRunner,
        metrics_logger,
        episode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        info: Dict[str, Any] = _last_info(episode)

        success = 1.0 if bool(info.get("goal_reached", False)) else 0.0
        fallen = 1.0 if bool(info.get("is_fallen", False)) else 0.0
        final_x = float(info.get("x_position", 0.0))
        final_y = float(info.get("y_position", 0.0))
        final_yaw = float(info.get("unicycle_yaw", 0.0))

        metrics_logger.log_value("success", success)
        metrics_logger.log_value("is_fallen", fallen)
        metrics_logger.log_value("final_x_position", final_x)
        metrics_logger.log_value("final_y_position", final_y)
        metrics_logger.log_value("final_yaw", final_yaw)

        for k in ["reward_shaping", "reward_alive_bonus", "reward_goal_bonus", "reward_fall_penalty"]:
            v = info.get(k, None)
            if v is not None:
                metrics_logger.log_value(k, float(v))

    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs) -> None:
        envr = result.get(ENV_RUNNER_RESULTS, {})
        cm_fallback = result.get("custom_metrics", {})

        episode_return_mean = float(envr.get("episode_return_mean", 0.0))
        episodes_lifetime = int(envr.get("num_episodes_lifetime", 0))
        steps_lifetime = int(envr.get("num_env_steps_sampled_lifetime", 0))

        final_x_mean = _extract_mean(envr, "final_x_position", fallback=cm_fallback)
        final_y_mean = _extract_mean(envr, "final_y_position", fallback=cm_fallback)
        final_yaw_mean = _extract_mean(envr, "final_yaw", fallback=cm_fallback)
        success_rate = 100.0 * _extract_mean(envr, "success", fallback=cm_fallback)
        fall_rate = 100.0 * _extract_mean(envr, "is_fallen", fallback=cm_fallback)

        iteration = int(result.get("training_iteration", 0))
        print(
            f"\n[Training] Iteration: {iteration} | "
            f"Episodes: {episodes_lifetime} | Steps: {steps_lifetime:,} | "
            f"Episode_Reward: {episode_return_mean:.2f} | "
            f"Final_X_Position: {final_x_mean:.2f}m | "
            f"Final_Y_Position: {final_y_mean:.2f}m | "
            f"Final_Yaw_Angle: {final_yaw_mean:+.3f} rad | "
            f"Success_Rate: {success_rate:.1f}% | Fall_Rate: {fall_rate:.1f}%"
        )

        eval_block = result.get(EVALUATION_RESULTS, {})  # = result["evaluation"]
        eval_envr = eval_block.get(ENV_RUNNER_RESULTS, {})
        eval_episodes = int(eval_envr.get("num_episodes", 0))

        if eval_episodes > 0:
            eval_return_mean = float(eval_envr.get("episode_return_mean", 0.0))

            eval_final_x_mean = _extract_mean(eval_envr, "final_x_position")
            eval_success_rate = 100.0 * _extract_mean(eval_envr, "success")
            eval_fall_rate = 100.0 * _extract_mean(eval_envr, "is_fallen")

            print(
                f"[Evaluation] Episodes: {eval_episodes} | "
                f"Episode_Reward: {eval_return_mean:.2f} | "
                f"Final_X_Position: {eval_final_x_mean:.2f}m | "
                f"Success_Rate: {eval_success_rate:.1f}% | "
                f"Fall_Rate: {eval_fall_rate:.1f}%"
            )

            target_success = 0.95
            if eval_success_rate >= target_success * 100:
                print(
                    f"[Evaluation] Target Success Rate Achieved! "
                    f"({eval_success_rate:.1f}% >= {target_success*100:.0f}%)"
                )
            else:
                print(
                    f"[Evaluation] Target not yet reached "
                    f"({eval_success_rate:.1f}% < {target_success*100:.0f}%)"
                )


class EvalSuccessStopper(Stopper):
    def __init__(
        self,
        target_success_rate: float,
        minimum_number_of_episodes: int,
    ):
        self.target_success_rate = float(target_success_rate)
        self.minimum_number_of_episodes = int(minimum_number_of_episodes)

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        eval_block = result.get(EVALUATION_RESULTS, {})
        eval_envr = eval_block.get(ENV_RUNNER_RESULTS, {})
        eval_episodes = int(eval_envr.get("num_episodes", 0))

        success_val = eval_envr.get("success", None)
        if success_val is None:
            cm = eval_block.get("custom_metrics", {})
            success_val = cm.get("success_mean", None)
        if success_val is None:
            return False

        if isinstance(success_val, dict):
            eval_success_rate = float(success_val.get("mean", 0.0))
        else:
            eval_success_rate = float(success_val)

        should_stop = (
            eval_episodes >= self.minimum_number_of_episodes
            and eval_success_rate >= self.target_success_rate
        )

        if should_stop:
            print("\n" + "=" * 80)
            print("[Training Complete] Successfully reached target success rate")
            print(
                f"Final Success Rate: {eval_success_rate * 100:.1f}% "
                f"(Target: {self.target_success_rate * 100:.0f}%)"
            )
            print(f"Total Evaluation Episodes: {eval_episodes}")
            print("=" * 80 + "\n")

        return should_stop

    def stop_all(self) -> bool:
        return False