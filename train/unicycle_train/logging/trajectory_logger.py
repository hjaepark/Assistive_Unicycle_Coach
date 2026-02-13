# train/unicycle_train/logging/trajectory_logger.py

import csv
import time
import numpy as np
from pathlib import Path
import gymnasium as gym

TRAJECTORY_FIELDS = [
    "participant_id","condition","phase","timestamp","episode","num_steps",
    *[f"obs_{i}" for i in range(19)],
    "unicycle_roll","unicycle_pitch","pendulum_ang",
    "human_action_0","human_action_1",
    "coach_action_0","coach_action_1",
    "weight_assist","effective_intervention_norm",
    "final_action_0","final_action_1","final_action_2","final_action_3",
    "reward","done","truncated",
    "num_success","success_indicator","is_fallen","skill_level",
]

class TrajectoryLoggerWrapper(gym.Wrapper):
    """
    Log RL student training in user study format (step-level CSV)
    Uses human_action instead of student_action for data analysis.
    """
    def __init__(self, env: gym.Env, out_csv: Path,
                 participant_id: str = "RL",
                 condition: str = "without-coach",
                 phase: str = "training"):
        super().__init__(env)
        self.participant_id = participant_id
        self.condition = condition
        self.phase = phase

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(out_csv, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=TRAJECTORY_FIELDS)
        self._writer.writeheader()

        self._t0 = time.time()
        self._episode = 1
        self._global_step = 0
        self._need_ep_inc = False

        self._last_num_success = 0
        self.skill_level = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self._need_ep_inc:
            self._episode += 1
            self._need_ep_inc = False

        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        self._global_step += 1

        # Check for success
        num_success = int(info.get("num_success", 0))
        success_indicator = (num_success > self._last_num_success)
        if success_indicator:
            self._last_num_success = num_success

        # Build row
        obs = np.asarray(obs).reshape(-1)
        row = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "phase": self.phase,
            "timestamp": time.time() - self._t0,
            "episode": self._episode,
            "num_steps": self._global_step,
        }
        
        # Observations
        for i in range(19):
            row[f"obs_{i}"] = float(obs[i])

        row["unicycle_roll"]  = float(info.get("unicycle_roll", 0.0))
        row["unicycle_pitch"] = float(info.get("unicycle_pitch", 0.0))
        row["pendulum_ang"]   = float(info.get("pendulum_ang", float(obs[7])))

        # Student action (no coach)
        a = np.asarray(action).reshape(-1)
        row["human_action_0"] = float(a[0])
        row["human_action_1"] = float(a[1])

        row["coach_action_0"] = 0.0
        row["coach_action_1"] = 0.0
        row["weight_assist"] = 0.0
        row["effective_intervention_norm"] = 0.0

        row["final_action_0"] = float(a[0])
        row["final_action_1"] = float(a[1])
        row["final_action_2"] = 0.0
        row["final_action_3"] = 0.0

        row["reward"] = float(reward)
        row["done"] = bool(done)
        row["truncated"] = bool(truncated)

        row["num_success"] = num_success
        row["success_indicator"] = bool(success_indicator)
        row["is_fallen"] = bool(info.get("is_fallen", False))
        row["skill_level"] = float(self.skill_level)

        self._writer.writerow(row)

        if done or truncated:
            self._need_ep_inc = True

        return obs, reward, done, truncated, info

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
        return super().close()