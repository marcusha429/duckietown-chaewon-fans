#!/bin/bash
#SBATCH --job-name=PPO_8_envs_default           # Set the name of the job
#SBATCH --output=job_logs/%j.log                # Specify where to save the job's standard output and error logs. 
#SBATCH --time=6:00:00                          # Set the maximum run time for the job
#SBATCH --mem=64GB                              # Allocate memory for the job.
#SBATCH --cpus-per-task=16                      # Allocate CPU cores for the task.
#SBATCH --partition=standard                    # Specify the partition to use.

# Set the Pyglet window to headless
export PYGLET_NO_LIB=1

# Run the Python training script
python3 train.py --config configs/model1.yaml