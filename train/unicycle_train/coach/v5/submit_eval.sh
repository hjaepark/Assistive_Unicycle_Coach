#!/bin/bash
# train/unicycle_train/coach/v5/submit_eval.sh
#
# Evaluate V5 trajectory-latent coach across 100 held-out seeds (0-99) on CPU.
# Each seed runs as an independent SLURM job; all 100 jobs run in parallel
# across PACE nodes, so wall clock time equals single-seed evaluation time.
#
# Usage: from the v5 directory:
#     bash submit_eval.sh
#
# Note: this script CALLS sbatch internally for each seed, so we run it
# with `bash`, not `sbatch`. The sbatch calls happen inside the for-loop.

PYTHON=/storage/scratch1/7/hpark459/envs/unienv/bin/python
PROJECT=/storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach

# V5 coach checkpoint to evaluate (binary intervention {0, 1}, obs_dim=34)
PI_PATH=${PROJECT}/train/unicycle_train/coach/v5/logs/v5_20260526_162913_pool100/models/pi_I_iter_0100.pth

LOG_DIR=${PROJECT}/train/unicycle_train/coach/v5/logs_slurm
mkdir -p ${LOG_DIR}

# 100 evaluation seeds (0 through 99), disjoint from training seeds (1000-1099)
SEEDS=$(seq 0 99)

for SEED in ${SEEDS}; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=v5_eval_s${SEED}
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/v5_eval100_s${SEED}_%j.out
#SBATCH --error=${LOG_DIR}/v5_eval100_s${SEED}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hpark459@gatech.edu

export PATH=/storage/scratch1/7/hpark459/envs/unienv/bin:\$PATH
export PYTHONPATH=
export PYTHONHOME=

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd ${PROJECT}

${PYTHON} -m train.unicycle_train.coach.v5.eval \\
    --mode both \\
    --pi-path ${PI_PATH} \\
    --seeds ${SEED} \\
    --device cpu \\
    --tag eval100seeds_pool100_iter100
EOF
    echo "Submitted V5 evaluation seed ${SEED}"
done

echo ""
echo "All 100 V5 evaluation jobs submitted on CPU."
echo "Monitor with: squeue -u hpark459 | grep v5_eval"
echo "Logs: ${LOG_DIR}"
