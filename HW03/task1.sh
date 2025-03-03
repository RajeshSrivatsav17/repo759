#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW03_task1
#SBATCH -c 2
#SBATCH -o task1_output_%a.txt
#SBATCH -e task1_error_%a.txt
#SBATCH --array=1-20

# Compute n for this specific task
t=$((2**$SLURM_ARRAY_TASK_ID))
n=$((2**10))
echo "./task1 $SLURM_ARRAY_TASK_ID"
./task1 $n $t