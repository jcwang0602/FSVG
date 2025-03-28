# A Simple and Better Baseline for Visual Grounding

### Install
```
conda create -n fsvg Python=3.10
conda activate fsvg
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Data Preparation
You can download the images from the original source and place them in `./image_data` folder:
- RefCOCO/RefCOCO+/RefCOCOg
- Flickr30K Entities
- Visual Genome

Finally, the `./image_data` folder will have the following structure:

```angular2html
|-- ln_data
   |-- flickr30k
   |-- mscoco/images/train2014/
   |-- visual-genome
```

### Training

Training on ReferIt. 
```
GPUS=4
MODEL='ViT-B/16' # ViT-B/16 ViT-L/14
DATA_NAME=gref_umd # unc unc+ gref gref_umd referit
DATA_ROOT='/share/wangjingchao/vg_data/image_data'
SPLIT_ROOT='/share/wangjingchao/vg_data/split_data'

OUTPUT=outputs/${DATA_NAME}/fsvg_vitb16_fs

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 27763 --use_env train.py \
--aug_crop --aug_scale --aug_translate \
--model $MODEL \
--fs \
--dataset ${DATA_NAME} \
--data_root ${DATA_ROOT} \
--split_root ${SPLIT_ROOT} \
--output_dir ${OUTPUT}
```
Please refer to [train.sh](train.sh) for training commands on other datasets.

### Evaluation

Download the model weights from [Baidu Netdisk](https://pan.baidu.com/s/1fqXEvflaC2O9-46gg0SSEA?pwd=qt2h).

Evaluation on ReferIt. 
```
GPUS=4
MODEL='ViT-B/16' # ViT-B/16 ViT-L/14
DATA_NAME=referit # unc unc+ gref gref_umd referit
DATA_ROOT='/share/wangjingchao/vg_data/image_data'
SPLIT_ROOT='/share/wangjingchao/vg_data/split_data'

OUTPUT=outputs/${DATA_NAME}/fsvg_vitb16_fs

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 27763 --use_env eval.py \
--model $MODEL \
--dataset $DATA_NAME \
--fs \
--data_root ${DATA_ROOT} \
--split_root ${SPLIT_ROOT} \
--eval_model ${OUTPUT}/best_checkpoint.pth  \
--eval_set val \
--output_dir ${OUTPUT}
```
Please refer to [eval.sh](eval.sh) for eval commands on other datasets.

<!-- ### Our checkpoints

Our checkpoints are available at [Baidu Netdisk](). -->

## Acknowledgement

Our model is related to [CLIP-VG](https://github.com/linhuixiao/CLIP-VG), [OSTrack](https://github.com/botaoye/OSTrack). Thanks for their great work!

<!-- ## Citation
If our work is useful for your research, please consider cite:
```
@misc{fsvg,
      title={A Simple and Better Baseline for Visual Grounding}, 
      author={Wangjingchao and Yuyang Tang and Wenfei Yang and Tianzhu Zhang and Jinpeng Zhang and Mengxue Kang},
      year={2025},
      eprint={2401.11228},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->

<!-- [//]: # (## Star History)

[//]: # ()
[//]: # ([![Star History Chart]&#40;https://api.star-history.com/svg?repos=jcwang0602/PLVL&type=Date&#41;]&#40;https://star-history.com/#linhuixiao/HiVG&Date&#41;) -->