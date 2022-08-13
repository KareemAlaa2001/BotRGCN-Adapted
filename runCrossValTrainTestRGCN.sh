#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N run_crossValTrainTest
#$ -cwd
#$ -l h_rt=47:00:00
#$ -l h_vmem=48G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 47 hours: -l h_rt
#  memory limit of 48 Gbyte (dissa ram heavy boi): -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# load anaconda
module load anaconda

# Activate botRGCN conda environment
source activate botRGCN

# Run the python file for trainTestHeteroMinibatched.py
python CrossValTrainTestRGCN.py --augmented_dataset $1 --dataset_variant $2   