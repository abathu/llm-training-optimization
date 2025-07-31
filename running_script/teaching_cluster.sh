#!/bin/bash

#SBATCH --job-name=llm 
#SBATCH --output=/home/s2678328/MasterProjectCode/logs/slurm-%j.out
#SBATCH --error=/home/s2678328/MasterProjectCode/logs/slurm-%j.out




echo "Start Training "

export HF_HOME=/home/s2678328/.cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# === 3. Run your Python script ===
python /home/s2678328/llm-training-optimization/employ_model.py


echo "End of training"