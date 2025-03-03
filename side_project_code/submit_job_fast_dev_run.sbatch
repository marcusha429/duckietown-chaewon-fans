#!/bin/bash
#SBATCH --job-name=fast_dev_run         # Set the name of the job (used for tracking in the job scheduler).
#SBATCH --output=job_logs/job_output_%j.log  # Specify where to save the job's standard output and error logs. 
                                            # The %j is replaced with the job ID.
#SBATCH --time=00:10:00                 # Set the maximum run time for the job (10 minutes in this case).
#SBATCH --mem=64GB                      # Allocate 64GB of memory for the job.
#SBATCH --cpus-per-task=16              # Allocate 16 CPU cores for the task.
#SBATCH --partition=standard            # Specify the partition to use.

# Set the Pyglet window to headless
export PYGLET_NO_LIB=1

# Run the Python training script
python3 train.py --config configs/fast_dev_run.yaml