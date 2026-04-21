#!/bin/bash
# Submit multi-student generalization experiment
# Tests if V3.1 coach (trained on seed 42) generalizes to new students
# Usage: bash submit_multistudent_eval.sh

mkdir -p logs

PYTHON=/storage/scratch1/7/hpark459/envs/unienv/bin/python
PROJECT=/storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach

# V3.1 coach checkpoint (trained on seed 42 student)
PI_PATH=/storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach/train/unicycle_train/coach/v3/logs/v3_20260404_145317/models/pi_I_final.pth

for SEED in 500 600 700 800 900; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=multi_s${SEED}
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi_s${SEED}_%j.out
#SBATCH --error=logs/multi_s${SEED}_%j.err

export PATH=/storage/scratch1/7/hpark459/envs/unienv/bin:\$PATH
export PYTHONPATH=
export PYTHONHOME=

cd ${PROJECT}
${PYTHON} -m train.unicycle_train.coach.v3.eval \\
    --mode both \\
    --pi-path ${PI_PATH} \\
    --seeds ${SEED}
EOF
    echo "Submitted seed ${SEED}"
done