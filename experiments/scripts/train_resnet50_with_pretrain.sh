#!/usr/bin/env bash
# Usage:
# ./experiments/scripts/train.sh GPU [options args to {train,test}_net.py]
#
# Example:
# ./experiments/scripts/train.sh 0 \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=resnet50
DATASET=psdb

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  psdb)
    TRAIN_IMDB="psdb_train"
    TEST_IMDB="psdb_test"
    PT_DIR="psdb"
    ITERS=100000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${DATASET}_train_${NET}_softmax_with_pretrain.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

mpirun -n 8 python2 tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/solver.prototxt \
  --weights output/psdb_pretrain/resnet50_softmax/resnet50_iter_8000.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/train_resnet50.yml \
  --rand \
  --set EXP_DIR resnet50_softmax_with_pretrain \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
