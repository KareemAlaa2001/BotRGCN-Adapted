#!/bin/sh
qsub runCrossValTrainTestRGCN.sh --cross_val_mode --no_augment 0
qsub runCrossValTrainTestRGCN.sh --cross_val_mode --no_augment 1
qsub runCrossValTrainTestRGCN.sh --cross_val_mode --augment 0
qsub runCrossValTrainTestRGCN.sh --cross_val_mode --augment 1

qsub runCrossValTrainTestRGCN.sh --test_mode --no_augment 0
qsub runCrossValTrainTestRGCN.sh --test_mode --no_augment 1
qsub runCrossValTrainTestRGCN.sh --test_mode --augment 0
qsub runCrossValTrainTestRGCN.sh --test_mode --augment 1
