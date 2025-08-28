# train/unicycle_train/evaluate.py

from typing import Any, Dict, Optional

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.env_runner import EnvRunner
from ray.tune.stopper import Stopper


def get_last_info_from_episode(episode) -> Dict[str, Any]:
    agent_ids = list(episode.get_agents())
    first_agent_id = agent_ids[0] if agent_ids else None
    return episode.last_info_for(first_agent_id) if first_agent_id is not None else {}


class SuccessCallback(RLlibCallback):
    def on_episode_end(
        self,
        *,
        env_runner: EnvRunner,
        episode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        info: Dict[str, Any] = get_last_info_from_episode(episode)
        
        episode.custom_metrics["success"] = 1.0 if bool(info.get("goal_reached", False)) else 0.0
        
        episode.custom_metrics["final_x_position"] = float(info.get("x_position", 0.0))
        
        episode.custom_metrics["is_fallen"] = 1.0 if bool(info.get("is_fallen", False)) else 0.0

    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs) -> None:
        if ("custom_metrics" in result) and ("success_mean" in result["custom_metrics"]):
            training_success_mean = float(result["custom_metrics"]["success_mean"])
            result.setdefault("env_runners", {})["success_rate"] = training_success_mean

        if "evaluation" in result:
            evaluation_block: Dict[str, Any] = result["evaluation"]
            if ("custom_metrics" in evaluation_block) and ("success_mean" in evaluation_block["custom_metrics"]):
                evaluation_success_mean = float(evaluation_block["custom_metrics"]["success_mean"])
                evaluation_block.setdefault("env_runners", {})["success_rate"] = evaluation_success_mean
        
        iteration = result.get("training_iteration", 0)
        
        if "env_runners" in result:
            env_data = result["env_runners"]
            total_episodes = env_data.get("num_episodes_lifetime", 0)
            total_steps = env_data.get("num_env_steps_sampled_lifetime", 0)
            episode_reward_mean = env_data.get("episode_return_mean", 0.0)
            
            final_x_mean = 0.0
            success_rate = 0.0
            fall_rate = 0.0
            
            if "custom_metrics" in result:
                custom = result["custom_metrics"]
                final_x_mean = custom.get("final_x_position_mean", 0.0)
                success_rate = custom.get("success_mean", 0.0) * 100
                fall_rate = custom.get("is_fallen_mean", 0.0) * 100
            
            print(f"\n[Training] Iteration: {iteration} | Episodes: {total_episodes} | "
                  f"Steps: {total_steps:,} | Episode_Reward: {episode_reward_mean:.2f} | "
                  f"Final_X_Distance: {final_x_mean:.2f}m | Success_Rate: {success_rate:.1f}% | "
                  f"Fall_Rate: {fall_rate:.1f}%")
        
        if "evaluation" in result:
            eval_block = result["evaluation"]
            if "env_runners" in eval_block:
                eval_data = eval_block["env_runners"]
                eval_episodes = eval_data.get("num_episodes", 0)
                eval_reward_mean = eval_data.get("episode_return_mean", 0.0)
                
                eval_final_x_mean = 0.0
                eval_success_rate = 0.0
                eval_fall_rate = 0.0
                
                if "custom_metrics" in eval_block:
                    eval_custom = eval_block["custom_metrics"]
                    eval_final_x_mean = eval_custom.get("final_x_position_mean", 0.0)
                    eval_success_rate = eval_custom.get("success_mean", 0.0) * 100
                    eval_fall_rate = eval_custom.get("is_fallen_mean", 0.0) * 100
                
                print(f"[Evaluation] Episodes: {eval_episodes} | "
                      f"Episode_Reward: {eval_reward_mean:.2f} | "
                      f"Final_X_Distance: {eval_final_x_mean:.2f}m | "
                      f"Success_Rate: {eval_success_rate:.1f}% | "
                      f"Fall_Rate: {eval_fall_rate:.1f}%")
                
                target_success = 0.95
                if eval_episodes >= 100:
                    if eval_success_rate >= target_success * 100:
                        print(f"[Evaluation] Target Success Rate Achieved! ({eval_success_rate:.1f}% >= {target_success*100:.0f}%)")
                    else:
                        print(f"[Evaluation] Target not yet reached ({eval_success_rate:.1f}% < {target_success*100:.0f}%)")


class EvalSuccessStopper(Stopper):
    def __init__(
        self,
        target_success_rate: float,
        minimum_number_of_episodes: int,
    ):
        self.target_success_rate = float(target_success_rate)
        self.minimum_number_of_episodes = int(minimum_number_of_episodes)

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        if "evaluation" not in result:
            return False

        evaluation_block: Dict[str, Any] = result["evaluation"]

        if ("custom_metrics" not in evaluation_block) or ("success_mean" not in evaluation_block["custom_metrics"]):
            return False
        if ("env_runners" not in evaluation_block) or ("num_episodes" not in evaluation_block["env_runners"]):
            return False

        evaluation_success_rate = float(evaluation_block["custom_metrics"]["success_mean"])
        evaluation_number_of_episodes = int(evaluation_block["env_runners"]["num_episodes"])

        should_stop = (
            evaluation_number_of_episodes >= self.minimum_number_of_episodes
            and evaluation_success_rate >= self.target_success_rate
        )
        
        if should_stop:
            print(f"\n{'='*80}")
            print(f"[Training Complete] Successfully reached target success rate")
            print(f"Final Success Rate: {evaluation_success_rate * 100:.1f}% (Target: {self.target_success_rate * 100:.0f}%)")
            print(f"Total Evaluation Episodes: {evaluation_number_of_episodes}")
            print(f"{'='*80}\n")
        
        return should_stop

    def stop_all(self) -> bool:
        return False