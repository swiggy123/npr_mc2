#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/eval_base_out.log
#SBATCH --err=slurm/eval_base_err.log
#SBATCH --job-name="LMQG Eval Base"
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m pip install -e ./lm-question-generation

NB_PATH="eval_base.ipynb"
python3 -m papermill $NB_PATH $NB_PATH -k 'python3'