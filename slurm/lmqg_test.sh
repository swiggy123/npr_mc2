#!/bin/sh
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=slurm/log_out.txt
#SBATCH --err=slurm/log_err.txt
#SBATCH --job-name="LMQG Inference Test"
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m papermill LMQG_Inference_Test.ipynb LMQG_Inference_Test.ipynb -k 'python3'