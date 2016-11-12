#!/usr/bin/env bash

CAFFE=caffe
DATASET=psdb
NET=resnet50
SNAPSHOTS_DIR=output/${DATASET}_pretrain

LOG="experiments/logs/${DATASET}_pretrain_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

cd $(dirname ${BASH_SOURCE[0]})/../../

mkdir -p ${SNAPSHOTS_DIR}

GLOG_logtostderr=1 mpirun -n 8 ${CAFFE}/build/tools/caffe train \
  -solver models/${DATASET}/${NET}/pretrain_solver.prototxt \
  -weights data/imagenet_models/${NET}.caffemodel 2>&1 | tee ${LOG}