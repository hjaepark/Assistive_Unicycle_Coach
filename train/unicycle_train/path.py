# train/unicycle_train/path.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDENT_DIR = PROJECT_ROOT / "train" / "unicycle_train" / "student"
LOGS_STUDENT_DIR = STUDENT_DIR / "logs_student"

DATA_STUDENT_DIR = STUDENT_DIR / "data"

HSE_DIR = PROJECT_ROOT / "hse"
HSE_DATA_DIR = PROJECT_ROOT / "data"

for _dir in (STUDENT_DIR, LOGS_STUDENT_DIR, DATA_STUDENT_DIR, HSE_DIR, HSE_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)