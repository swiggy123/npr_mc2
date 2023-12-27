#!/bin/sh
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/logs/train_small_combined_out.txt
#SBATCH --err=slurm/logs/train_small_combined_err.txt
#SBATCH --job-name="LMQG Train Small Combined"
#SBATCH --exclude=gpu22a,gpu22b,node15,sdas2
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m pip install -e ./lm-question-generation

lmqg-train-search -d "StellarMilk/squad_newsqa" -m "t5-small" -b 2 -g 2 4 -c "small_combined_trained_ckpt" -i 'paragraph' -o 'questions_answers' -p 'qag' --epoch-partial 10 -e 15 --max-length-output-eval 512 --max-length-output 512