#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW05_task1
#SBATCH -c 1
#SBATCH -o task1_output.txt
#SBATCH -e task1_error.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
./task1