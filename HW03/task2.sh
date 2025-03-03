#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW03_task2
#SBATCH -c 20
#SBATCH -o task2_output_%a.txt
#SBATCH -e task2_error_%a.txt
#SBATCH --array=1-20

# Compute n for this specific task
t=$(($SLURM_ARRAY_TASK_ID))
n=$((2**10))
echo "./task2 $SLURM_ARRAY_TASK_ID"
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2_$SLURM_ARRAY_TASK_ID -fopenmp
./task2_$SLURM_ARRAY_TASK_ID $n $t