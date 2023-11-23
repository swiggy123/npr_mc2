#!/bin/sh
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=slurm/log_out.txt
#SBATCH --err=slurm/log_err.txt
#SBATCH --job-name="Axolotl Inference Test"
python3 -m pip install -r requirements.txt

cd axolotl
python3 -m pip install packaging
python3 -m pip install -e '.[flash-attn,deepspeed]'
echo ""

python3 -m axolotl.cli.inference examples/openllama-3b/lora.yml --lora_model_dir="./lora-out"
echo "Done!"