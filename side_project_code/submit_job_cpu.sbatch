#!/bin/bash
#SBATCH -A cs175_class                          # Account to charge
#SBATCH --job-name=duckietown_lane_follower           # Set the name of the job
#SBATCH --output=job_logs/duckietown_lane_follower-%j.log                # Specify where to save the job's standard output and error logs. 
#SBATCH --time=16:00:00                          # Set the maximum run time for the job
#SBATCH --mem=64GB                              # Allocate memory for the job.
#SBATCH --cpus-per-task=16                      # Allocate CPU cores for the task.
#SBATCH --partition=standard                    # Specify the partition to use.

# Set the Pyglet window to headless
export PYGLET_NO_LIB=1

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python training script
python3 -u train.py --config configs/myconfig.yaml