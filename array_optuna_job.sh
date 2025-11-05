#!/bin/bash
#SBATCH -J param_sweep
#SBATCH --account=gts-skousik3-paid
#SBATCH --array=0-49
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -q inferno
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
echo "Worker ID: $SLURM_ARRAY_TASK_ID / 49"
echo "Cores: 16"
echo "Memory: 64GB"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================"

python -m train.unicycle_train.cotraining.tune

echo "================================"
echo "Worker $SLURM_ARRAY_TASK_ID DONE at $(date)"
echo "================================"