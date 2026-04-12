#!/bin/bash
# Submit all 3 V3.1 coach training jobs to PACE
# Usage: bash submit_all.sh

mkdir -p logs

PYTHON=/storage/scratch1/7/hpark459/envs/unienv/bin/python
PROJECT=/storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach

for SEED in 100 200 300; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=v3_seed${SEED}
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v3_seed${SEED}_%j.out
#SBATCH --error=logs/v3_seed${SEED}_%j.err

export PATH=/storage/scratch1/7/hpark459/envs/unienv/bin:\$PATH
export PYTHONPATH=
export PYTHONHOME=

cd ${PROJECT}
${PYTHON} -m train.unicycle_train.coach.v3.train --seed ${SEED}
EOF
    echo "Submitted seed ${SEED}"
done