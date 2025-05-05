# train/unicycle_train/create_trained_model.py
# RLlib 2.12.0 – legacy(“old”) stack

import os
import datetime
import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode


# 1) Environment ----------------------------------------------------------
def unicycle_env_creator(env_config):
    """
    Gymnasium-compatible factory for the custom unicycle environment.
    """
    import gymnasium as gym
    import envs.unicycle  # Registers Unicycle-v0
    return gym.make("Unicycle-v0", **env_config)


tune.register_env("UnicycleRLlib", unicycle_env_creator)


# 2) Callbacks ------------------------------------------------------------
TARGET_SUCCESS = 200


class MetricAndStopCallback(DefaultCallbacks):
    """
    • Log custom metrics (successes, falls, final X-position)  
    • Implement early stopping once TARGET_SUCCESS is reached
    """

    # ---------- per-episode ------------------------------------------------
    def on_episode_end(self, *, episode: Episode, **__) -> None:
        last_info = episode.last_info_for()
        if last_info is None:            # Guard against abnormal termination
            return

        # Custom metrics
        episode.custom_metrics["num_success"] = last_info.get("num_success", 0)
        episode.custom_metrics["num_falls"]   = last_info.get("num_falls",   0)
        episode.custom_metrics["x_final"]     = last_info.get("x_position",  0)

    # ---------- per-training-iteration ------------------------------------
    def on_train_result(self, *, result: dict, **__) -> None:
        succ  = result["custom_metrics"].get("num_success_max", 0)
        falls = result["custom_metrics"].get("num_falls_max",   0)
        dist  = result["custom_metrics"].get("x_final_mean",    0)
        it    = result["training_iteration"]
        rew   = result["episode_reward_mean"]

        print(f"[it={it:>4}] reward={rew:8.2f}  "
              f"success={succ}  falls={falls}  x̄={dist:5.2f}")

        if succ >= TARGET_SUCCESS:
            print(f"[Stop] Reached {TARGET_SUCCESS} successes → terminating.")
            result["done"] = True


# 3) SAC configuration ----------------------------------------------------
cfg = (
    SACConfig()                         # still a SACConfig object
        .environment("UnicycleRLlib")
        .framework("torch")
        .resources(num_gpus=1)
        .callbacks(MetricAndStopCallback)
).to_dict()                             # keep a pure-dict for Tune


# 4) Launch experiment ----------------------------------------------------
def main() -> None:
    logs_root = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_root, exist_ok=True)
    run_name = f"unicycle_student_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    tuner = tune.Tuner(
        "SAC",
        param_space=cfg,
        run_config=RunConfig(
            name=run_name,
            storage_path=logs_root,
            stop={"timesteps_total": 3_000_000},
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end           = True,
                checkpoint_frequency        = 1,
                checkpoint_score_attribute  = "episode_reward_mean",
                checkpoint_score_order      = "max",
                num_to_keep                 = 2,
            ),
        ),
    )

    results = tuner.fit()
    best    = results.get_best_result()
    if best:
        print("Best checkpoint →", best.checkpoint.path)


if __name__ == "__main__":
    ray.shutdown()
    main()