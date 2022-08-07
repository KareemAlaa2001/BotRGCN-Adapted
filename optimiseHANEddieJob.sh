#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N optimiseHAN20
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=48G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 24 hours: -l h_rt
#  memory limit of 48 Gbyte (dissa ram heavy boi): -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# load anaconda
module load anaconda

# Activate botRGCN conda environment
source activate botRGCN

# Run the wandb agent for hyperparameter sweeping
wandb agent graphbois/test-project/gwi2p291