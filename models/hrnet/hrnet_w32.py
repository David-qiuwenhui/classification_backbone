# # ---------- 卷积模型的参数 ----------
model_cfg = dict(
    backbone=dict(type="HRNet", arch="w32"),
    neck=[
        dict(type="HRFuseScales", in_channels=(32, 64, 128, 256)),
        dict(type="GlobalAveragePooling"),
    ],
    head=dict(
        type="LinearClsHead",
        in_channels=2048,
        num_classes=5,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1, 5),
    ),
)

# ---------- 数据集加载的参数 ----------
# dataloader pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomResizedCrop", size=224, backend="pillow"),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(256, -1), backend="pillow"),
    dict(type="CenterCrop", crop_size=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

# ---------- 训练的参数 ----------
data_cfg = dict(
    batch_size=16,
    num_workers=8,
    train=dict(
        pretrained_flag=False,
        pretrained_weights="",
        freeze_flag=False,
        freeze_layers=("backbone",),
        epochs=10,
    ),
    test=dict(
        ckpt="./logs/HRNet/2023-01-18-20-00-31/Train_Epoch010-Loss0.901.pth",
        metrics=["accuracy", "precision", "recall", "f1_score", "confusion"],
        metric_options=dict(topk=(1, 5), thrs=None, average_mode="none"),
    ),
)

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        ".absolute_pos_embed": dict(decay_mult=0.0),
        ".relative_position_bias_table": dict(decay_mult=0.0),
    },
)

# batch 32
# lr = 0.1 *32 /256
# optimizer
# 训练优化器的参数
optimizer_cfg = dict(
    type="SGD", lr=0.1 * 16 / 256, momentum=0.9, weight_decay=1e-4
)  # 初始的学习率和batch size有关系

# learning
lr_config = dict(type="CosineAnnealingLrUpdater", min_lr=0)
