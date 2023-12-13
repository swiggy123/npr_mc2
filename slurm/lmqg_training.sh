#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/training_out.log
#SBATCH --err=slurm/training_err.log
#SBATCH --job-name="LMQG Train"
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

NB_PATH="training.ipynb"
python3 -m papermill $NB_PATH $NB_PATH -k 'python3'