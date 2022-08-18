#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N runHANFullHeteroWithConfig
#$ -cwd
#$ -l h_rt=47:00:00
#$ -l h_vmem=48G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 47 hours (just in case): -l h_rt
#  memory limit of 48 Gbyte (dissa ram heavy boi): -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# load anaconda
module load anaconda

# Activate botRGCN conda environment
source activate botRGCN

# Run the python file for trainTestHeteroMinibatched.py
python crossValWithConfigFile.py $1 $2 --dataset_variant $3 --hyperparamConfigFile $4