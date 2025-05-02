#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW07_t1
#SBATCH -c 1
#SBATCH -o task1_output_%a.txt
#SBATCH -e task1_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --array=5-14

# Compute n for this specific task
n=$((2**$SLURM_ARRAY_TASK_ID))
echo "./task1 $SLURM_ARRAY_TASK_ID"

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_$SLURM_ARRAY_TASK_ID
./task1_$SLURM_ARRAY_TASK_ID $n 1024