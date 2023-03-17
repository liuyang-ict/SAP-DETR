EXP_DIR=./outputs/r101-dc-50epoch/
if [ ! -d "outputs" ]; then
    mkdir outputs
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi


python -m torch.distributed.launch --nproc_per_node=8 \
  main.py -m sap_detr \
  --batch_size 1 \
  --dilation \
  --backbone resnet101 \
  --num_workers 10 \
  --epochs 50 \
  --lr_drop 40 \
  --enc_layers 6 \
  --dec_layers 6 \
  --transformer_activation relu \
  --num_select 300 \
  --num_queries 306 \
  --warmup_iters 1000 \
  --coco_path /dfs/data/Project/Datasets/coco \
  --meshgrid_refpoints_xy \
  --bbox_embed_diff_each_layer \
  --newconvinit \
  --sdg \
  --output_dir ${EXP_DIR} \
     2>&1 | tee ${EXP_DIR}/train.log
