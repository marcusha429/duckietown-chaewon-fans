#!/bin/bash
#SBATCH --job-name=train_fast_dev_run
#SBATCH --output=job_output_%j.log
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --partition=standard

# Set the Pyglet window to headless
export PYGLET_NO_LIB=1

# Run the Python training script
python3 train.py --config configs/fast_dev_run.yaml
# /usr/bin/xvfb-run python3 train.py --config configs/fast_dev_run.yaml # Use xvfb-run to use virtual display in headless mode