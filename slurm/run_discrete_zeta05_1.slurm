#!/bin/bash

#SBATCH --job-name=DM2

#SBATCH --gres=gpu:1 

#SBATCH --output=./DISCRETEzeta05_1MASK.out

#SBATCH --error=./DISCRETEzeta05_1MASK.err

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

srun -u python3 discrete.py -p 1 -m ppo_mask

