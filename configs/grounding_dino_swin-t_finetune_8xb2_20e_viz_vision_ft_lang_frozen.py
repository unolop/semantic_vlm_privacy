import json
from os import getenv

_base_ = ['./grounding_dino_swin-t_finetune_8xb2_20e_viz.py']

data_root = getenv(
    'VIZWIZ_DATA_ROOT',
    '/home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/',
)
load_from = getenv(
    'GROUNDING_DINO_CHECKPOINT',
    '/home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth',
)
annotation_path = data_root + 'base_annotations.json'
class_name = tuple(
    category['name']
    for category in sorted(
        json.loads(open(annotation_path).read())['categories'],
        key=lambda x: x['id'],
    )
)

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='base_annotations.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False),
    ),
)

val_dataloader = None
test_dataloader = None

val_evaluator = None
test_evaluator = None

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=50),
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1000)
val_cfg = None
test_cfg = None

# Keep the text encoder frozen to preserve open-vocabulary transfer,
# while adapting the visual detector to BLV-style image degradation.
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'language_model': dict(lr_mult=0.0),
            'backbone': dict(lr_mult=0.1),
        }
    ),
)
