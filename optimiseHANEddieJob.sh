#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N optimiseHAN              
#$ -cwd
#$ -l h_rt=10:00:00 
#$ -l h_vmem=32G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 10 hours: -l h_rt
#  memory limit of 32 Gbyte (dissa ram heavy boi): -l h_vmem

# load anaconda
module load anaconda

# Activate botRGCN conda environment
source activate botRGCN

# Run the wandb agent for hyperparameter sweeping
wandb agent graphbois/test-project/3p9u93j8