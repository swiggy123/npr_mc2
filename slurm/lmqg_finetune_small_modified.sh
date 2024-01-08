#!/bin/sh
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/logs/finetune_small_modified_out.txt
#SBATCH --err=slurm/logs/finetune_small_modified_err.txt
#SBATCH --job-name="LMQG Finetune Small Modified"
#SBATCH --exclude=gpu22a,gpu22b,node15,sdas2
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m pip install -e ./lm-question-generation

lmqg-train-search -d "StellarMilk/newsqa_modified" -m "lmqg/t5-small-squad-qag" -l 0.00001 --epoch-partial 5 -e 10 -b 2 -g 2 4 -c "small_finetuned_modified_ckpt" -i 'paragraph' -o 'questions_answers' -p 'qag' --max-length-output-eval 512 --max-length-output 512