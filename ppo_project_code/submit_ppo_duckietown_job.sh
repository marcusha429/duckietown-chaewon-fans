#!/bin/bash
#SBATCH -A cs175_class_gpu    ## Account to charge
#SBATCH --time=06:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes
#SBATCH --partition=gpu       ## Partition name
#SBATCH --mem=40GB            ## Allocated Memory
#SBATCH --cpus-per-task 8     ## Number of CPU cores
#SBATCH --gres=gpu:V100:1     ## Type and number of GPUs
#SBATCH --tmp=10GB            ## Add swap space

module load ffmpeg # necessary for saving gifs to tensorboard on HPC3
pip install psutil --user
python ducky_project.py