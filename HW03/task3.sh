#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW03_task3
#SBATCH -c 2
#SBATCH -o task3_output_%a.txt
#SBATCH -e task3_error_%a.txt
#SBATCH --array=1-10

# Compute n for this specific task
t=$((2**3))
n=$((2**10))
ts=$((2**SLURM_ARRAY_TASK_I))
echo "./task3 $SLURM_ARRAY_TASK_ID"
./task3 $n $t $ts