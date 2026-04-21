#!/bin/bash
# Submit multi-student generalization experiment
# Tests if V3.2 coach (trained on seed 42) generalizes to new students
# Usage: bash submit_multistudent_eval_2.sh
#        N_OBS=5 bash submit_multistudent_eval_2.sh

mkdir -p logs

PYTHON=/storage/scratch1/7/hpark459/envs/unienv/bin/python
PROJECT=/storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach
PI_PATH=/storage/home/hcoda1/7/hpark459/scratch/Assistive_Unicycle_Coach/train/unicycle_train/coach/v3_2/logs/v3_2_20260417_231103_seed42/models/pi_I_final.pth

N_OBS="${N_OBS:-20}"

for SEED in 500 600 700 800 900; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=multi2_s${SEED}_obs${N_OBS}
#SBATCH --account=gts-skousik3-paid
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi2_s${SEED}_obs${N_OBS}_%j.out
#SBATCH --error=logs/multi2_s${SEED}_obs${N_OBS}_%j.err

export PATH=/storage/scratch1/7/hpark459/envs/unienv/bin:\$PATH
export PYTHONPATH=
export PYTHONHOME=

cd ${PROJECT}
${PYTHON} -m train.unicycle_train.coach.v3_2.eval \\
    --mode both \\
    --pi-path ${PI_PATH} \\
    --seeds ${SEED} \\
    --n-observation-iters ${N_OBS}
EOF
    echo "Submitted seed ${SEED} (n_obs=${N_OBS})"
done