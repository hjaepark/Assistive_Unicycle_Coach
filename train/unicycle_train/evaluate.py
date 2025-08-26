# train/unicycle_train/evaluate.py

import math
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.tune.stopper import Stopper

# Wilson 95% Confidence Interval
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1.0 + (z*z)/n
    center = p + (z*z)/(2*n)
    rad = z * math.sqrt((p*(1-p) + (z*z)/(4*n)) / n)
    lb = (center - rad) / denom
    ub = (center + rad) / denom
    return p, max(0.0, lb), min(1.0, ub)

class SuccessCallback(RLlibCallback):

    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["reached"] = False
        episode.custom_data["fell"] = False

    def on_episode_step(self, *, episode, **kwargs):
        info = None
        try:
            info = episode.get_infos(-1)
        except Exception:
            info = None
        if isinstance(info, dict) and info:
            if info.get("goal_reached", False):
                episode.custom_data["reached"] = True
            if info.get("is_fallen", False):
                episode.custom_data["fell"] = True

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        reached = bool(episode.custom_data.get("reached", False))
        fell = bool(episode.custom_data.get("fell", False))
        success = 1.0 if (reached and not fell) else 0.0

        metrics_logger.log_value("success_rate", success, reduce="mean")


    def on_evaluate_end(self, *, evaluation_metrics, **kwargs):
        envr = evaluation_metrics.setdefault("env_runners", {})
        p = float(envr.get("success_rate", 0.0))
        n = int(envr.get("num_episodes", 0))
        k = int(round(p * n))
        envr["success_n"] = n
        _, lb, ub = wilson_ci(k, n)
        envr["success_ci_lb"] = lb
        envr["success_ci_ub"] = ub

class WilsonSuccessStopper(Stopper):
    def __init__(self, min_lb: float = 0.90, consec: int = 2, max_iters: int = 2000):
        self.min_lb = float(min_lb)
        self.consec = int(consec)
        self.max_iters = int(max_iters)
        self.streak = {}

    def __call__(self, trial_id, result):
        if result.get("training_iteration", 0) >= self.max_iters:
            return True
        
        envr = result.get("evaluation", {}).get("env_runners", {})
        lb = envr.get("success_ci_lb")
        n  = envr.get("success_n")
        if lb is None or not n:
            return False
        passed = float(lb) >= self.min_lb
        s = self.streak.get(trial_id, 0)
        self.streak[trial_id] = s + 1 if passed else 0
        return self.streak[trial_id] >= self.consec

    def stop_all(self):
        return False