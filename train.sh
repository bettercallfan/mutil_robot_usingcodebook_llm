#!/bin/bash
#SBATCH -J vq6
#SBATCH -p gpu
#SBATCH --gres=gpu:V100:1
#SBATCH -w GPU248
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

cd /nfsdat1/home/rfchenslm/centurymaze
mkdir -p logs

ENV_PATH=/nfsdat1/home/rfchenslm/.conda/envs/centurymaze
PYTHON_BIN=$ENV_PATH/bin/python

export PATH="$ENV_PATH/bin:${PATH}"
export LD_LIBRARY_PATH="$ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/nfsdat1/home/rfchenslm/centurymaze/isaacgym/python:${PYTHONPATH:-}"

echo "===== ENV CHECK ====="
date
hostname
nvidia-smi
echo "python in PATH: $(which python)"
python -V
$PYTHON_BIN -V
$PYTHON_BIN -c "import isaacgym; print('isaacgym ok')"

echo "===== START TRAIN ====="
date

$PYTHON_BIN /nfsdat1/home/rfchenslm/centurymaze/src/mazebots/session.py \
  --ctrl_mode 2 \
  --headless 1 \
  --model_name vq6 \
  --n_envs 9 \
  --n_bots 112 \
  --use_vq_comm 1 \
  --vq_weight 0.001 \
  --vq_commitment_cost 0.15 \
  --vq_num_embeddings 4096 \
  --vq_codebook_lr 2e-5 
echo "===== TRAIN DONE ====="
date