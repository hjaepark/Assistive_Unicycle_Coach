# train/unicycle_train/monitor_callback/coach_monitor.py

from stable_baselines3.common.callbacks import BaseCallback

class CoachMonitor(BaseCallback):
    """
    Logs coach-related info to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self._logger = None

    def _init_callback(self) -> None:
        super()._init_callback()
        if self.model is not None:
            self._logger = self.model.logger

    def _on_step(self) -> bool:
        if self._logger is None:
            return True
        infos = self.locals["infos"]
        action_names = ["wheel_motor", "roll_control", "pitch_control", "yaw_control"]
        for info in infos:
            if "coach_action" in info:
                for i, val in enumerate(info["coach_action"]):
                    self._logger.record(f"coach/{action_names[i]}", val)
            if "coach_action_prime" in info:
                for i, val in enumerate(info["coach_action_prime"]):
                    self._logger.record(f"coach/prime_{action_names[i]}", val)
        return True