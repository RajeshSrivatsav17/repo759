#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW05_task3
#SBATCH -c 1
#SBATCH -o task3_output_%a.txt
#SBATCH -e task3_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --array=10-29

# Compute n for this specific task
n=$((2**$SLURM_ARRAY_TASK_ID))
echo "./task3 $SLURM_ARRAY_TASK_ID"
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3_$SLURM_ARRAY_TASK_ID
./task3_$SLURM_ARRAY_TASK_ID $n