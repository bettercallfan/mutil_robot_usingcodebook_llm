#!/bin/bash
#SBATCH -J nollmonlymlp_v_qwen1_env1_robot8
#SBATCH -p gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --exclude=GPU41
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

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

$PYTHON_BIN /nfsdat1/home/rfchenslm/centurymaze/src/mazebots0/session.py \
  --ctrl_mode 2 \
  --headless 1 \
  --model_name nollmonlymlp_v_qwen1_env1_robot8 \
  --n_envs 1 \
  --n_bots 8 \
  --batch_size 8 \
  --llm_bridge_enabled 0

echo "===== TRAIN DONE ====="
date
