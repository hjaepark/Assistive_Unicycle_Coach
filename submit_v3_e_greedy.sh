sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=v3_eps50
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v3_eps50_%j.out
#SBATCH --error=logs/v3_eps50_%j.err

export PATH=/storage/scratch1/7/hpark459/envs/unienv/bin:$PATH
export PYTHONPATH=
export PYTHONHOME=

cd /storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach
/storage/scratch1/7/hpark459/envs/unienv/bin/python -m train.unicycle_train.coach.v3.train --seed 100 --config train/unicycle_train/coach/v3/configs/epsilon_greedy_decay50.yaml
SLURM