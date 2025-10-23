#!/bin/bash
#SBATCH -J param_sweep                       # Job name
#SBATCH --account=gts-skousik3-paid          # Paid account
#SBATCH --array=0-9                          # 10 workers
#SBATCH -N 1                                 # 1 node per worker
#SBATCH -n 16                                # 16 cores per worker
#SBATCH --mem=64G                            # 64GB memory
#SBATCH -t 48:00:00                          # 48 hours
#SBATCH -q inferno                           # QOS
#SBATCH -o tune_logs/optuna_%A_%a.out
#SBATCH -e tune_logs/optuna_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hpark459@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p tune_logs

source ~/.bashrc
conda activate unienv

echo "================================"
echo "Parameter Sweep"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Worker ID: $SLURM_ARRAY_TASK_ID / 9"
echo "Cores: 16"
echo "Memory: 64GB"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================"

# Each worker 50 trials
# 10 workers × 50 trials = 500 trials!
python -m train.unicycle_train.cotraining.tune 50

echo "================================"
echo "Worker $SLURM_ARRAY_TASK_ID DONE at $(date)"
echo "================================"
