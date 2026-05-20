#!/bin/bash
#SBATCH -J codebook_kv384zhibiao_new2
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

$PYTHON_BIN runner.py --version proto --spec eval_perf --args "--model_name codebook_kv384_new2 --vq_num_embeddings 384 --vq_embedding_dim 96"

echo "===== TRAIN DONE ====="
date