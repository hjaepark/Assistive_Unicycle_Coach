# train/unicycle_train/path.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDENT_DIR = PROJECT_ROOT / "train" / "unicycle_train" / "student"
LOGS_STUDENT_DIR = STUDENT_DIR / "logs_student"


for _dir in (STUDENT_DIR, LOGS_STUDENT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)