#!/bin/bash

#SBATCH --job-name=DM1.1

#SBATCH --gres=gpu:1 

#SBATCH --output=./DISCRETEzeta0_05MASK2.out

#SBATCH --error=./DISCRETEzeta0_05MASK2.err

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

srun -u python3 discrete.py -p 0 -m ppo_mask

