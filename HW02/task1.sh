#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW02_task1
#SBATCH -c 2
#SBATCH -o task1_output_%a.txt
#SBATCH -e task1_error_%a.txt
#SBATCH --array=10-30

# Compute n for this specific task
n=$((2**SLURM_ARRAY_TASK_ID))

echo "./task1 $SLURM_ARRAY_TASK_ID"
./task1 $n