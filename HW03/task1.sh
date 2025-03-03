#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW03_task1
#SBATCH -c 20
#SBATCH -o task1_output_%a.txt
#SBATCH -e task1_error_%a.txt
#SBATCH --array=1-20

# Compute n for this specific task
t=$(($SLURM_ARRAY_TASK_ID))
n=$((2**10))
echo "./task1 $SLURM_ARRAY_TASK_ID"
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1_$SLURM_ARRAY_TASK_ID -fopenmp
./task1_$SLURM_ARRAY_TASK_ID $n $t