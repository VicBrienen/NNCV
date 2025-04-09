#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:05:00

# Run inside container, including diagnostics
srun apptainer exec --nv --env-file .env container.sif /bin/bash -c "
    echo 'CUDA_VISIBLE_DEVICES: '\$CUDA_VISIBLE_DEVICES
    echo 'Running nvidia-smi inside container...'
    nvidia-smi
    echo 'Starting training script...'
    /bin/bash main.sh
"