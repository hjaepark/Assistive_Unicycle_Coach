# train/unicycle_train/path.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS_DIR      = PROJECT_ROOT / "experiments"
LOGS_STUDENT_DIR     = PROJECT_ROOT / "logs_student"
LOGS_COACH_DIR       = PROJECT_ROOT / "logs_coach"

STUDENT_DIR          = PROJECT_ROOT / "train" / "unicycle_train" / "student"
COACH_DIR            = PROJECT_ROOT / "train" / "unicycle_train" / "coach"

EXPERT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] /
    "train/unicycle_train/coach/models/unicycle_expert" /
    "SAC_UnicycleStudentRLlib_60938_00000_0_2025-05-11_03-13-02" /
    "checkpoint_004225"
)

for _dir in (EXPERIMENTS_DIR, LOGS_STUDENT_DIR, LOGS_COACH_DIR, STUDENT_DIR, COACH_DIR):
    _dir.mkdir(parents=True, exist_ok=True)