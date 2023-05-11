#!/bin/bash

#SBATCH --job-name=MASKR2

#SBATCH --gres=gpu:1 

#SBATCH --output=./maskr2.out

#SBATCH --error=./maskr2.err

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

srun -u python3 discrete2.py

