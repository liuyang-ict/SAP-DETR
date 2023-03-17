EXP_DIR=./outputs/r50-50epoch/
if [ ! -d "outputs" ]; then
    mkdir outputs
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi


python -m torch.distributed.launch --nproc_per_node=1 \
  main.py -m sap_detr \
  --batch_size 4 \
  --backbone resnet50 \
  --num_workers 6 \
  --epochs 36 \
  --lr_drop 30 \
  --enc_layers 6 \
  --dec_layers 6 \
  --transformer_activation relu \
  --num_select 300 \
  --num_queries 306 \
  --coco_path /dfs/data/Project/Datasets/coco \
  --meshgrid_refpoints_xy \
  --bbox_embed_diff_each_layer \
  --newconvinit \
  --sdg \
  --resume ${EXP_DIR}/checkpoint0049.pth \
  --output_dir ${EXP_DIR} \
  --eval
