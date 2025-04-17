# train/unicycle_train/monitor_callback/student_monitor.py

from stable_baselines3.common.callbacks import BaseCallback

class StudentMonitor(BaseCallback):
    """
    Logs student-related info to TensorBoard.
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
        for info in infos:
            if "reward" in info:
                self._logger.record("student/reward", info["reward"])
            if "student_action" in info:
                action_names = ["wheel_motor", "roll_control", "pitch_control", "yaw_control"]
                for i, val in enumerate(info["student_action"]):
                    self._logger.record(f"student/{action_names[i]}", val)
            if "unicycle_roll" in info:
                self._logger.record("student/unicycle_roll", info["unicycle_roll"])
            if "unicycle_pitch" in info:
                self._logger.record("student/unicycle_pitch", info["unicycle_pitch"])
            if "x_position" in info:
                self._logger.record("student/x_position", info["x_position"])
            if "y_position" in info:
                self._logger.record("student/y_position", info["y_position"])
            if "num_success" in info:
                self._logger.record("student/num_success", info["num_success"])
            if "num_falls" in info:
                self._logger.record("student/num_falls", info["num_falls"])
            if "is_fallen" in info:
                self._logger.record("student/is_fallen", float(info["is_fallen"]))
        return True
