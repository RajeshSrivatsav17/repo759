#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW02_task2
#SBATCH -c 2
#SBATCH -o task2_output_%a.txt
#SBATCH -e task2_error_%a.txt

./task2 400 300