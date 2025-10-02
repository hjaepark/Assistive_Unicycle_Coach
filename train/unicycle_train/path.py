# train/unicycle_train/path.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDENT_DIR = PROJECT_ROOT / "train" / "unicycle_train" / "student"
LOGS_STUDENT_DIR = STUDENT_DIR / "logs_student"

# offline dataset (save runs for estimator)
DATA_STUDENT_DIR = STUDENT_DIR / "data"

COACH_DIR = PROJECT_ROOT / "train" / "unicycle_train" / "coach"
COACH_MODELS_DIR = COACH_DIR / "models"

STB_ARTIFACTS_DIR = COACH_DIR / "estimators" / "stb_artifacts" / "2_layer+previous_reward+128_hidden_dim"

for _dir in (STUDENT_DIR, LOGS_STUDENT_DIR, DATA_STUDENT_DIR, COACH_DIR, COACH_MODELS_DIR, STB_ARTIFACTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)