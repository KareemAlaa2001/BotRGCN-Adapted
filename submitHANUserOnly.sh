#!/bin/sh
qsub submitCrossValWithConfig.sh --cross_val_mode --no_augment 0 config/nonAugWinnerConfig.json
qsub submitCrossValWithConfig.sh --cross_val_mode --no_augment 1 config/nonAugWinnerConfig.json

qsub submitCrossValWithConfig.sh --test_mode --no_augment 0 config/nonAugWinnerConfig.json
qsub submitCrossValWithConfig.sh --test_mode --no_augment 1 config/nonAugWinnerConfig.json