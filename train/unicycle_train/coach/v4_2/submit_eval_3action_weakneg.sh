#!/bin/bash
# train/unicycle_train/coach/v4_2/submit_eval_3action_weakneg.sh
#
# Evaluate the V4.2 3-action weak-negative coach across 100 held-out seeds (0-99) on CPU.
# 3-action weak-negative means w_t can take values in {-0.5, 0, 1}.
# Each seed runs as an independent SLURM job, all 100 in parallel.
#
# Before running: verify PI_PATH points to the correct training run and iteration.
#
# Usage from this directory:
#     bash submit_eval_3action_weakneg.sh

PYTHON=/storage/scratch1/7/hpark459/envs/unienv/bin/python
PROJECT=/storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach

# V4.2 3-action weak-negative coach checkpoint
# Timestamp 130846 corresponds to the 3-action weak-negative training run
PI_PATH=${PROJECT}/train/unicycle_train/coach/v4_2/logs/v4_1_20260505_130846_pool100/models/pi_I_iter_0100.pth

# Absolute log directory path so the script works regardless of where sbatch runs
LOG_DIR=${PROJECT}/train/unicycle_train/coach/v4_2/logs_slurm
mkdir -p ${LOG_DIR}

# 100 evaluation seeds, disjoint from the 1000-1099 training seed pool
SEEDS=$(seq 0 99)

for SEED in ${SEEDS}; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=v42_3w_s${SEED}
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/v42_3w_s${SEED}_%j.out
#SBATCH --error=${LOG_DIR}/v42_3w_s${SEED}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hpark459@gatech.edu

# Reset Python environment for clean unienv activation
export PATH=/storage/scratch1/7/hpark459/envs/unienv/bin:\$PATH
export PYTHONPATH=
export PYTHONHOME=

# Cap thread count to match cpus-per-task (prevents oversubscription with 100 parallel jobs)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd ${PROJECT}

${PYTHON} -m train.unicycle_train.coach.v4_2.eval \\
    --mode both \\
    --pi-path ${PI_PATH} \\
    --seeds ${SEED} \\
    --device cpu \\
    --tag 3action_weakneg_iter100
EOF
    echo "Submitted V4.2 3-action weak-negative seed ${SEED}"
done

echo ""
echo "All 100 V4.2 3-action weak-negative evaluation jobs submitted."
echo "Monitor: squeue -u hpark459 | grep v42_3w"
echo "Logs:    ${LOG_DIR}"