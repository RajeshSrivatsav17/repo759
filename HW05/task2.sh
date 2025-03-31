#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW05_task2
#SBATCH -c 1
#SBATCH -o task2_output.txt
#SBATCH -e task2_error.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2
./task2