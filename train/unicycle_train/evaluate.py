# train/unicycle_train/evaluate.py

from typing import Any, Dict

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.tune.stopper import Stopper

class SuccessCallback(DefaultCallbacks):

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        self.total_episodes = 0

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Any,
        env_index: int,
        **kwargs,
    ):
        info = episode.last_info_for()
        if not info:
            info = base_env.get_sub_environments()[env_index].last_info

        if info:
            episode.custom_metrics["success"] = 1.0 if bool(info.get("goal_reached", False)) else 0.0
            episode.custom_metrics["fall"] = 1.0 if bool(info.get("is_fallen", False)) else 0.0
            episode.custom_metrics["final_x_position"] = float(info.get("x_position", 0.0))
            episode.custom_metrics["final_y_position"] = float(info.get("y_position", 0.0))
            episode.custom_metrics["final_yaw"] = float(info.get("unicycle_yaw", 0.0))

    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs) -> None:
        training_metrics_source = result.get("env_runners", result)
        cm = training_metrics_source.get("custom_metrics", {})
        
        episodes_this_iter = training_metrics_source.get("episodes_this_iter", 0)
        self.total_episodes += episodes_this_iter
        
        episodes_lifetime = self.total_episodes
        steps_lifetime = result.get("timesteps_total", 0)
        episode_return_mean = training_metrics_source.get("episode_reward_mean", float("nan"))

        final_x_mean = cm.get("final_x_position_mean", 0.0)
        final_y_mean = cm.get("final_y_position_mean", 0.0)
        final_yaw_mean = cm.get("final_yaw_mean", 0.0)
        success_rate = 100.0 * cm.get("success_mean", 0.0)
        fall_rate = 100.0 * cm.get("fall_mean", 0.0)
        
        iteration = result.get("training_iteration", 0)
        print(
            f"\n[Training] Iteration: {iteration} | "
            f"Episodes: {episodes_lifetime} | Steps: {steps_lifetime:,} | "
            f"Episode_Reward: {episode_return_mean:.2f} | "
            f"Final_X_Position: {final_x_mean:.2f}m | "
            f"Final_Y_Position: {final_y_mean:.2f}m | "
            f"Final_Yaw_Angle: {final_yaw_mean:+.3f} rad | "
            f"Success_Rate: {success_rate:.1f}% | Fall_Rate: {fall_rate:.1f}%"
        )

        eval_block = result.get("evaluation", {})
        if eval_block:
            eval_cm = eval_block.get("custom_metrics", eval_block)
            eval_episodes = eval_block.get("episodes_this_iter", 0)
            eval_return_mean = eval_block.get("episode_reward_mean", float("nan"))
            
            eval_final_x_mean = eval_cm.get("final_x_position_mean", 0.0)
            eval_success_rate = 100.0 * eval_cm.get("success_mean", 0.0)
            eval_fall_rate = 100.0 * eval_cm.get("fall_mean", 0.0)
            
            if eval_episodes > 0:
                print(
                    f"[Evaluation] Episodes: {eval_episodes} | "
                    f"Episode_Reward: {eval_return_mean:.2f} | "
                    f"Final_X_Position: {eval_final_x_mean:.2f}m | "
                    f"Success_Rate: {eval_success_rate:.1f}% | "
                    f"Fall_Rate: {eval_fall_rate:.1f}%"
                )

class EvalSuccessStopper(Stopper):
    def __init__(self, target_success_rate: float, minimum_number_of_episodes: int):
        self.target_success_rate = float(target_success_rate)
        self.minimum_number_of_episodes = int(minimum_number_of_episodes)

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        eval_block = result.get("evaluation")
        if not eval_block:
            return False

        eval_cm = eval_block.get("custom_metrics", eval_block)
        eval_episodes = eval_block.get("episodes_this_iter", 0)
        eval_success_rate = eval_cm.get("success_mean", 0.0)

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