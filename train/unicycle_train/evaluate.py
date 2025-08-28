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

    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs) -> None:
        if ("custom_metrics" in result) and ("success_mean" in result["custom_metrics"]):
            training_success_mean = float(result["custom_metrics"]["success_mean"])
            result.setdefault("env_runners", {})["success_rate"] = training_success_mean

        if "evaluation" in result:
            evaluation_block: Dict[str, Any] = result["evaluation"]
            if ("custom_metrics" in evaluation_block) and ("success_mean" in evaluation_block["custom_metrics"]):
                evaluation_success_mean = float(evaluation_block["custom_metrics"]["success_mean"])
                evaluation_block.setdefault("env_runners", {})["success_rate"] = evaluation_success_mean


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

        return (
            evaluation_number_of_episodes >= self.minimum_number_of_episodes
            and evaluation_success_rate >= self.target_success_rate
        )

    def stop_all(self) -> bool:
        return False
