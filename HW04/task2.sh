#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW03_task2
#SBATCH -c 8
#SBATCH -o task2_output.txt
#SBATCH -e task2_error.txt

# Compute n for this specific task
echo "./task2 $SLURM_ARRAY_TASK_ID"
g++ task2.cpp -Wall -O3 -std=c++17 -o task2
./task2 100 5