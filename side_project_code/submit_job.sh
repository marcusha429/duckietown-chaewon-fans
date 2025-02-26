#!/bin/bash
#SBATCH --job-name=train_ppo                # Name of the job
#SBATCH --output=train_ppo_output.txt       # Output file for stdout
#SBATCH --error=train_ppo_error.txt         # Error file for stderr
#SBATCH --time=01:00:00                     # Maximum runtime (1 hour)
#SBATCH --mem=16G                           # RAM required (16 GB in this case)
#SBATCH --cpus-per-task=4                   # Number of CPUs per task (4 in this case)
#SBATCH --partition=general                 # Specify the partition (change as per your cluster's setup)

# Activate the virtual environment or set up the environment for the job
module load conda

# Activate conda environment
conda activate duckietown

# Run the Python script with the specified config
python3 train_ppo.py --config configs/model1.yaml
