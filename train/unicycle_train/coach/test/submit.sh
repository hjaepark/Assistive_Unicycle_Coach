#!/bin/bash
#SBATCH --job-name=buf_verify
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=cpu-small
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=train/unicycle_train/coach/test/logs_slurm/verify_%A_%a.out

# Buffer verification: ONE student (seed 1, strong, collapsed in V5),
# mislabel vs honest, repeated over 5 learning seeds each.
#   2 modes x 5 reps = 10 tasks
# If honest consistently beats mislabel on post-removal -> buffer is the cause.
STUDENT=1
MODES=(mislabel honest)
REPS=(101 102 103 104 105)

MODE_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
REP_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
MODE=${MODES[$MODE_IDX]}
REP=${REPS[$REP_IDX]}

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate unienv
mkdir -p train/unicycle_train/coach/test/logs_slurm

echo "Task ${SLURM_ARRAY_TASK_ID}: student=${STUDENT} mode=${MODE} rep=${REP}"
python -m train.unicycle_train.coach.test.run_diagnosis \
    --seed ${STUDENT} --rep-seed ${REP} \
    --intervene-rate 1.0 --buffer-mode ${MODE} \
    --train-episodes 500 --eval-episodes 100 --device cpu \
    --out-dir train/unicycle_train/coach/test/results_verify