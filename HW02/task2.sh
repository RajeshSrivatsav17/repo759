#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW02_task2
#SBATCH -c 2
#SBATCH -o task2_output.txt
#SBATCH -e task2_error.txt

./task2 400 299