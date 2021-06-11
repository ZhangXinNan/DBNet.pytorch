
CUDA_VISIBLE_DEVICES=0 python3 tools/train_zx.py \
    --config_file config/icdar2015_resnet50_FPN_DBhead_polyLR.ICDAR2019-LSVT.yaml

tensorboard --logdir output/DBNet_resnet50_FPN_DBHead