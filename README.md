
## Introduction

PFG DINO is a prompt-free vision-language model for environmental perception
in automated driving systems.

## Installation

PFG DINO is built on [MMetection](https://github.com/open-mmlab/mmdetection).

Please refer to [Installation](https://github.com/open-mmlab/mmdetection/docs/en/get_started.md/#Installation) for installation instructions.

## Getting Started

Please see [get_started.md](https://github.com/open-mmlab/mmdetection/docs/en/get_started.md) for the basic usage of MMDetection.


## Training PFG DINO:

### Preparation:
Chang the config:

```shell
configs/PF_grounding_dino/cityscapes_local_detection.py
```
and set 
```shell
data_root = 'Path_to_your_Cityscapes_dataset'
```

Then, Chang the config:

```shell
configs/PF_grounding_dino/PF_grounding_dino_swin-t_finetune_b2_12e_cityscapes.py
```
and set 
```shell
model = dict(
    preset_text_prompts_memory=dict(
        save_prompt_path=r'Path_to_the_text_prompt_memory',
    ),
    bbox_head=dict(num_classes=num_classes)
)
```
NOTE:
The text prompt memory is available in [提取码: jeey](https://pan.baidu.com/s/1E2GYbKfZO_lbnxfvC5muzA?pwd=jeey)

"PF_grounding_dino_swin-t_finetune_b2_12e_cityscapes_metainfo_auto6_wo_inference_MultiPosCrossEntropyLoss\save_prompts.pth"

### Start the training

```shell
python tools/train.py \
    configs/PF_grounding_dino/TPF_grounding_dino_swin-t_finetune_b2_12e_cityscapes.py 
```


## Model Zoo

dataset: Cityscaspes

| Model            | Backbone   | mAP(%) | Download                                                              |
|------------------|------------|--------|-----------------------------------------------------------------------|
| 'YOLOv8'         | CSPDarkNet | 45.4   |                                                                       |
| `Grounding DINO` | Swin-T     | 48.9   |                                                                       |
| `PFG DINO`       | Swin-T     | 49.8   | [提取码: jeey](https://pan.baidu.com/s/1E2GYbKfZO_lbnxfvC5muzA?pwd=jeey) |

