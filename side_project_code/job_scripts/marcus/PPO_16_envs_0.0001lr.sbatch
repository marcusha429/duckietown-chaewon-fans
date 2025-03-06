#!/bin/bash
#SBATCH --job-name=PPO_16_envs_0.0001lr             # Job name
#SBATCH -A cs175_class_gpu                 # class GPU
#SBATCH --output=job_logs/PPO_16_envs_0.0001lr-%j.log
#SBATCH --nodes=1
#SBATCH --time=35:00:00                    # Maximum run time
#SBATCH --mem=64GB                         # RAM allocation
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --partition=gpu                    # Partition name
#SBATCH --gres=gpu:V100:1                  # GPU allocation

# Set the Pyglet window to headless
export PYGLET_NO_LIB=1

# Run the Python training script
python3 train.py --config configs/PPO_16_envs_0.0001lr.yaml
