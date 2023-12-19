#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/logs/push_to_hf_out.txt
#SBATCH --err=slurm/logs/push_to_hf_err.txt
#SBATCH --job-name="LMQG push to HF"
#SBATCH --exclude=gpu23a,gpu23b,gpu23c,gpu23d
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m pip install -e ./lm-question-generation

#lmqg-push-to-hf -m "small_finetuned_ckpt/best_model" -a "t5-small-newsqa-qag-finetuned" -o "StellarMilk" --use-auth-token
lmqg-push-to-hf -m "base_trained_ckpt/best_model" -a "t5-base-newsqa-qag-trained" -o "StellarMilk" --use-auth-token