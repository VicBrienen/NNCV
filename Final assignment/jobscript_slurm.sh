#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=10:00:00

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh