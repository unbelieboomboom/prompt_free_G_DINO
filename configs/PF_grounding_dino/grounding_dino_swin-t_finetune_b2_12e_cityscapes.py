_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_cityscapes.py'

data_root = 'G:/datasets/cityscapes_used/'
class_name = ('person', 'car', 'truck', 'rider', 'bicycle', 'motorcycle', 'bus', 'train', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

# train_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='annotations/trainval.json',
#         data_prefix=dict(img='images/')))

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            metainfo=metainfo)))

# val_dataloader = dict(
#     dataset=dict(
#         metainfo=metainfo,
#         data_root=data_root,
#         ann_file='annotations/test.json',
#         data_prefix=dict(img='images/')))

val_dataloader = dict(
        dataset=dict(
            metainfo=metainfo))

test_dataloader = val_dataloader

# val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
# test_evaluator = val_evaluator

max_epoch = 12

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=1))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

# auto_scale_lr = dict(base_batch_size=16)
auto_scale_lr = dict(enable=True, base_batch_size=6)
