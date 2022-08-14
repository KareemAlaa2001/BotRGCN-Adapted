#!/bin/sh
qsub runCrossValParameterised.sh --cross_val_mode --no_augment 0
qsub runCrossValParameterised.sh --cross_val_mode --no_augment 1
qsub runCrossValParameterised.sh --cross_val_mode --augment 0 
qsub runCrossValParameterised.sh --cross_val_mode --augment 1
qsub runCrossValParameterised.sh --cross_val_mode --augment 2

qsub runCrossValParameterised.sh --test_mode --no_augment 0
qsub runCrossValParameterised.sh --test_mode --no_augment 1
qsub runCrossValParameterised.sh --test_mode --augment 0
qsub runCrossValParameterised.sh --test_mode --augment 1
qsub runCrossValParameterised.sh --test_mode --augment 2