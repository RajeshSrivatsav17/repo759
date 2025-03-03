#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW03_task3
#SBATCH -c 20
#SBATCH -o task3_output_%a.txt
#SBATCH -e task3_error_%a.txt
#SBATCH --array=1-10

# Compute n for this specific task
t=$((2**3))
n=$((10**6))
ts=$((2**$SLURM_ARRAY_TASK_ID))
echo "./task3 $SLURM_ARRAY_TASK_ID"
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3_$SLURM_ARRAY_TASK_ID -fopenmp
./task3_$SLURM_ARRAY_TASK_ID $n $t $ts