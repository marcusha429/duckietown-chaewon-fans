#!/bin/bash
#SBATCH --job-name=duckietown_lane_follower             # Job name
#SBATCH -A cs175_class_gpu            # class GPU
#SBATCH --output=job_logs/SAC_lr0.0001_buf1000000_ls4096_bs256_tau0.005_gamma0.98_tf1_gs1_ent0.05-%j.log
#SBATCH --nodes=1
#SBATCH --time=16:00:00               # Maximum run time
#SBATCH --mem=64GB                    # RAM allocation
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --partition=gpu               # Partition name
#SBATCH --gres=gpu:V100:1             # GPU allocation

# Set the Pyglet window to headless
export PYGLET_NO_LIB=1

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python training script
python3 -u train.py --config configs/myconfig.yaml