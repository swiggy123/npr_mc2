#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/logs/eval_base_out.txt
#SBATCH --err=slurm/logs/eval_base_err.txt
#SBATCH --job-name="LMQG Eval Base"
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m pip install -e ./lm-question-generation

lmqg-eval -d "StellarMilk/newsqa" -m "lmqg/t5-base-squad-qag" -e "evaluation_base/" -i "paragraph" -o "questions_answers" -l "en" --batch-size 2 --max-length-output 512 --max-length 512