GPUS=4
MODEL='ViT-B/16' # ViT-B/16 ViT-L/14
DATA_NAME=referit # unc unc+ gref gref_umd referit
DATA_ROOT='/share/wangjingchao/vg_data/image_data'
SPLIT_ROOT='/share/wangjingchao/vg_data/split_data'

# Train
# OUTPUT=outputs/${DATA_NAME}/fsvg_vitb16
# srun -p belt_road --gres=gpu:$GPUS --job-name=train --cpus-per-task=10 --quotatype=reserved \
#  python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 28601 --use_env train.py \
#  --aug_crop --aug_scale --aug_translate \
#  --model ${MODEL} \
#  --dataset ${DATA_NAME} \
#  --data_root ${DATA_ROOT} \
#  --split_root ${SPLIT_ROOT} \
#  --output_dir ${OUTPUT}

# Train w FS
OUTPUT=outputs/${DATA_NAME}/fsvg_vitb16_fs
srun -p belt_road --gres=gpu:$GPUS --job-name=b${DATA_NAME}${CE_KEEP} --quotatype=reserved \
 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 27763 --use_env train.py \
 --aug_crop --aug_scale --aug_translate \
 --model $MODEL \
 --fs \
 --dataset ${DATA_NAME} \
 --data_root ${DATA_ROOT} \
 --split_root ${SPLIT_ROOT} \
 --output_dir ${OUTPUT}
