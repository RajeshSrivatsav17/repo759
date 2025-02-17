#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J HW02_task3
#SBATCH -c 2
#SBATCH -o task3_output_%a.txt
#SBATCH -e task3_error_%a.txt

./task3