#!/bin/sh
qsub runCrossValTrainTestRGCN.sh False 0 False
qsub runCrossValTrainTestRGCN.sh False 1 False
qsub runCrossValTrainTestRGCN.sh True 0 False
qsub runCrossValTrainTestRGCN.sh True 1 False
qsub runCrossValTrainTestRGCN.sh False 0 True
qsub runCrossValTrainTestRGCN.sh False 1 True
qsub runCrossValTrainTestRGCN.sh True 0 True
qsub runCrossValTrainTestRGCN.sh True 1 True
