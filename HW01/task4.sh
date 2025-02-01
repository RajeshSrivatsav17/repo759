#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J FirstSlurm
#SBATCH -c 2
#SBATCH -o FirstSlurm.out 
#SBATCH -e FirstSlurm.err 
uname -n