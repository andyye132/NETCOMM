#!/bin/bash
#SBATCH -J nc_vod_validation
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -o /users/aiyer40/NETCOMM/results/logs/vod_validation_%j.out
mkdir -p /users/aiyer40/NETCOMM/results/logs
source /users/aiyer40/TRIAGE/.venv/bin/activate
cd /users/aiyer40/NETCOMM
python -u -m experiments.run_vod_validation
