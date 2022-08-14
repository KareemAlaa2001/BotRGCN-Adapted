#!/bin/sh
qsub runCrossValParameterised.sh False 0 False
qsub runCrossValParameterised.sh False 1 False
qsub runCrossValParameterised.sh True 0 False
qsub runCrossValParameterised.sh True 1 False
qsub runCrossValParameterised.sh True 2 False
qsub runCrossValParameterised.sh False 0 True
qsub runCrossValParameterised.sh False 1 True
qsub runCrossValParameterised.sh True 0 True
qsub runCrossValParameterised.sh True 1 True
qsub runCrossValParameterised.sh True 2 True