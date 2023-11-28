#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/log_out.txt
#SBATCH --err=slurm/log_err.txt
#SBATCH --job-name="LMQG Train"
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m pip install -e ./lm-question-generation

python3 -m papermill testing_lm_question_generation.ipynb testing_lm_question_generation.ipynb -k 'python3'