

pythoneate_filelist.py \
    --img_dir	/home/zhangxin/github/DBNet.pytorch/datasets/train/img \
    --gt_dir	/home/zhangxin/github/DBNet.pytorch/datasets/train/gt \
    --output	/home/zhangxin/github/DBNet.pytorch/datasets/train.txt


python create_filelist.py \
    --img_dir	/home/zhangxin/github/DBNet.pytorch/datasets/test/img \
    --gt_dir	/home/zhangxin/github/DBNet.pytorch/datasets/test/gt \
    --output	/home/zhangxin/github/DBNet.pytorch/datasets/test.txt



nohup python3 tools/train_zx.py \
    --config_file config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml \
    >nohup.20210608.icdar2015_resnet18_FPN_DBhead_polyLR.out &

current best, recall: 0.712349, precision: 0.908025, hmean: 0.798372, train_loss: 0.509826, best_model_epoch: 996.000000


tensorboard --logdir output/DBNet_resnet18_FPN_DBHead

python3 tools/train_zx.py \
    --config_file config/icdar2015_resnet50_FPN_DBhead_polyLR.yaml

tensorboard --logdir output/DBNet_resnet50_FPN_DBHead