# auto_experiments.py
# Hyper‑parameter sweep for the Unicycle SAC agent using Ray Tune
# Compatible with Ray/RLlib 2.12.0 (legacy trainer stack)

import os
import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode

# -----------------------------------------------------------
# 1)  Environment creator – same as baseline
# -----------------------------------------------------------

def unicycle_env_creator(env_config):
    import gymnasium as gym
    import envs.unicycle
    return gym.make("Unicycle-v0", **env_config)

tune.register_env("UnicycleRLlib", unicycle_env_creator)

# -----------------------------------------------------------
# 2)  Stop‑on‑success callback (10 cumulative successes)
# -----------------------------------------------------------
TARGET_SUCCESS = 10

class StopOnSuccessCallback(DefaultCallbacks):
    """Stop a trial once the *cumulative* success counter in the env hits `TARGET_SUCCESS`."""

    def on_episode_end(self, *, episode: Episode, **kwargs):
        # For single‑agent envs `last_info_for()` can be called without an arg.
        info = episode.last_info_for()
        if info is not None:
            episode.custom_metrics["num_success"] = info.get("num_success", 0)

    def on_train_result(self, *, result: dict, **kwargs):
        if result.get("custom_metrics", {}).get("num_success_max", 0) >= TARGET_SUCCESS:
            print("[Stop] total successes met → terminating trial.")
            # The trainer looks for this special key.
            result["done"] = True

# -----------------------------------------------------------
# 3)  Base SACConfig helper
# -----------------------------------------------------------

def build_base_config():
    return (
        SACConfig()
        .environment(env="UnicycleRLlib")
        .framework("torch")
        .resources(num_gpus=1)
        .debugging(seed=tune.randint(0, 2**32 - 1))
        .callbacks(StopOnSuccessCallback)
    )

# -----------------------------------------------------------
# 4)  Parameter search space
# -----------------------------------------------------------

def make_param_space():
    cfg = build_base_config().to_dict()

    # ---- top‑level grid/random params ----
    cfg.update(
        {
            "rollouts": {
                "num_rollout_workers": tune.grid_search([0, 4, 8]),
                "rollout_fragment_length": tune.grid_search([1, 50]),
            },
            "train_batch_size": tune.grid_search([256, 512]),
            "tau": tune.choice([0.005, 0.002]),
            "target_entropy": -6.0,  # float, not string!
        }
    )

    # ---- keep existing optimization dict and override subset ----
    opt = cfg.setdefault("optimization", {})
    opt.update(
        {
            "critic_learning_rate": tune.loguniform(3e-4, 1e-3),
            "entropy_learning_rate": tune.loguniform(1e-4, 3e-4),
            # actor_learning_rate remains at default 3e-4 (kept)
        }
    )

    return cfg

# -----------------------------------------------------------
# 5)  Main – launch sweep with ASHA
# -----------------------------------------------------------

def main():
    ray.init(log_to_driver=False)

    tuner = Tuner(
        "SAC",
        param_space=make_param_space(),
        run_config=RunConfig(
            name="unicycle_hparam_sweep",
            stop={"training_iteration": 3000},
            storage_path=os.path.join(os.getcwd(), "ray_logs"),
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=ASHAScheduler(
                time_attr="timesteps_total",
                max_t=5e5,
                grace_period=5e4,
                reduction_factor=3,
            ),
        ),
    )

    results = tuner.fit()
    if results.num_errors > 0:
        print(f"⚠️  {results.num_errors} trial(s) errored. Check ray_logs for details.")
    best = results.get_best_result()
    print("\n🏆 BEST trial reward_mean =", best.metrics.get("episode_reward_mean"))
    print("   config =", best.config)

if __name__ == "__main__":
    main()
