#!/bin/bash
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=cpu-small
#SBATCH --job-name=v4_2_4action_100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=train/unicycle_train/coach/v4_2/logs_slurm/v4_2_4action_%j.out
#SBATCH --error=train/unicycle_train/coach/v4_2/logs_slurm/v4_2_4action_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hpark459@gatech.edu

cd /storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach
source /storage/scratch1/7/hpark459/envs/unienv/bin/activate

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Memory:       $SLURM_MEM_PER_NODE MB"
echo "Start time:   $(date)"
echo "=========================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -u -m train.unicycle_train.coach.v4_2.train \
    --config train/unicycle_train/coach/v4_2/configs/4action.yaml

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="