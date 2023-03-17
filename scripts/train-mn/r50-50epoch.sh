#!/usr/bin/env bash

set -x

EXP_DIR=./outputs/r50-50epoch-mn/
PY_ARGS=${@:1}
if [ ! -d "outputs" ]; then
    mkdir outputs
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi


python -u \
  main.py -m sap_detr \
  --batch_size 4 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --backbone resnet50 \
  --num_workers 20 \
  --epochs 50 \
  --lr_drop 40 \
  --enc_layers 6 \
  --dec_layers 6 \
  --warmup_iters 1000 \
  --transformer_activation relu \
  --num_select 300 \
  --num_queries 306 \
  --coco_path /dfs/data/Project/Datasets/coco \
  --meshgrid_refpoints_xy \
  --bbox_embed_diff_each_layer \
  --sdg \
  --newconvinit \
  --output_dir ${EXP_DIR} \
  ${PY_ARGS}