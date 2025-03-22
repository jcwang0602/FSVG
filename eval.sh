GPUS=4
MODEL='ViT-B/16' # ViT-B/16 ViT-L/14
DATA_NAME=referit # unc unc+ gref gref_umd referit

DATA_ROOT='/share/wangjingchao/vg_data/image_data'
SPLIT_ROOT='/share/wangjingchao/vg_data/split_data'

# Eval FSVG
# OUTPUT=outputs/${DATA_NAME}/fsvg_vitb16
# srun -p belt_road --gres=gpu:$GPUS --job-name=$DATA_NAME --quotatype=reserved \
#  python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 28888 --use_env eval.py \
#  --model ${MODEL} \
#  --dataset ${DATA_NAME} \
#  --data_root ${DATA_ROOT} \
#  --split_root ${SPLIT_ROOT} \
#  --eval_model ${OUTPUT}/best_checkpoint.pth  \
#  --eval_set val \
#  --output_dir ${OUTPUT}

# Eval w FS
OUTPUT=outputs/${DATA_NAME}/fsvg_vitb16_fs
srun -p belt_road --gres=gpu:$GPUS --job-name=$DATA_NAME --quotatype=reserved \
 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 27763 --use_env eval.py \
 --model $MODEL \
 --dataset $DATA_NAME \
 --fs \
 --data_root ${DATA_ROOT} \
 --split_root ${SPLIT_ROOT} \
 --eval_model ${OUTPUT}/best_checkpoint.pth  \
 --eval_set val \
 --output_dir ${OUTPUT}