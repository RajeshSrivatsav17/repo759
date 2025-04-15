#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW06_task2
#SBATCH -c 1
#SBATCH -o task2_output_%a.txt
#SBATCH -e task2_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --array=10-29

# Compute n for this specific task
n=$((2**$SLURM_ARRAY_TASK_ID))
echo "./task2 $SLURM_ARRAY_TASK_ID"

nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2_$SLURM_ARRAY_TASK_ID
./task2_$SLURM_ARRAY_TASK_ID $n 128 1024