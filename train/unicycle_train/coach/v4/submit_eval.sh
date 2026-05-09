#!/bin/bash
# train/unicycle_train/coach/v4/submit_eval.sh
#
# Re-evaluate V4 binary coach across 100 held-out seeds (0-99) on CPU.
# Each seed runs as an independent SLURM job; all 100 jobs run in parallel
# across PACE nodes, so wall clock time equals single-seed evaluation time.
#
# Usage: from any directory (cd doesn't matter; PROJECT is absolute path):
#     sbatch /storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach/train/unicycle_train/coach/v4/submit_eval.sh
# Or simpler, from the v4 directory:
#     bash submit_eval.sh
#
# Note: this script CALLS sbatch internally for each seed, so we run it
# with `bash`, not `sbatch`. The sbatch calls happen inside the for-loop.

PYTHON=/storage/scratch1/7/hpark459/envs/unienv/bin/python
PROJECT=/storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach

# V4 coach checkpoint to evaluate (binary intervention {0, 1})
# Update this if you want to evaluate a different iteration's checkpoint
PI_PATH=${PROJECT}/train/unicycle_train/coach/v4/logs/v4_20260421_195950_pool100/models/pi_I_iter_0100.pth

# Where SLURM .out and .err logs go for each per-seed job
# Absolute path so it works regardless of where sbatch is called from
LOG_DIR=${PROJECT}/train/unicycle_train/coach/v4/logs_slurm
mkdir -p ${LOG_DIR}

# 100 evaluation seeds (0 through 99), disjoint from training seeds (1000-1099)
SEEDS=$(seq 0 99)

for SEED in ${SEEDS}; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=v4_eval_s${SEED}
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/v4_eval100_s${SEED}_%j.out
#SBATCH --error=${LOG_DIR}/v4_eval100_s${SEED}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hpark459@gatech.edu

# Reset Python environment to ensure clean activation of unienv
export PATH=/storage/scratch1/7/hpark459/envs/unienv/bin:\$PATH
export PYTHONPATH=
export PYTHONHOME=

# Limit thread counts to prevent oversubscription when 100 jobs run in parallel
# Without these limits, each job's PyTorch/numpy could spawn many threads,
# overloading the shared CPU pool on each node
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Move to project root so the python -m import path resolves correctly
cd ${PROJECT}

# Run V4 evaluation for this single seed
# --mode both runs both with-coach and no-coach baselines for direct comparison
${PYTHON} -m train.unicycle_train.coach.v4.eval \\
    --mode both \\
    --pi-path ${PI_PATH} \\
    --seeds ${SEED} \\
    --device cpu \\
    --tag eval100seeds_pool100_iter100
EOF
    echo "Submitted V4 evaluation seed ${SEED}"
done

echo ""
echo "All 100 V4 evaluation jobs submitted on CPU."
echo "Monitor with: squeue -u hpark459 | grep v4_eval"
echo "Logs: ${LOG_DIR}"