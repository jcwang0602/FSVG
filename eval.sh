GPUS=4
MODEL='ViT-B/16' # ViT-B/16 ViT-L/14
DATA_NAME=referit # unc unc+ gref gref_umd referit
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
OUTPUT=outputs/${DATA_NAME}/fsvg_vitb16_G4B32

srun -p mineru4s --gres=gpu:$GPUS --job-name=$DATA_NAME --quotatype=reserved \
 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 28888 --use_env eval.py \
 --model ${MODEL} \
 --dataset ${DATA_NAME} \
 --eval_model ${OUTPUT}/best_checkpoint.pth  \
 --eval_set val \
 --output_dir ${OUTPUT}

srun -p mineru4s --gres=gpu:$GPUS --job-name=$DATA_NAME --quotatype=reserved \
 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 28888 --use_env eval.py \
 --model ${MODEL} \
 --dataset ${DATA_NAME} \
 --eval_model ${OUTPUT}/best_checkpoint.pth  \
 --eval_set test \
 --output_dir ${OUTPUT}