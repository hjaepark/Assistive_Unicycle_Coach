sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=v3_2_train
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v3_2_train_%j.out
#SBATCH --error=logs/v3_2_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hpark459@gatech.edu

export PATH=/storage/scratch1/7/hpark459/envs/unienv/bin:$PATH
export PYTHONPATH=
export PYTHONHOME=

cd /storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach
/storage/scratch1/7/hpark459/envs/unienv/bin/python -m train.unicycle_train.coach.v3_2.train --seed 42
SLURM