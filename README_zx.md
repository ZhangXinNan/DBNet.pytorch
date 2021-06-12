
# train
```bash
python3 tools/train_zx.py --config_file config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml

# 2021-06-09 20:56:33,235 DBNet.pytorch INFO: current best, recall: 0.732203, precision: 0.892562, hmean: 0.804469, train_loss: 0.555731, best_model_epoch: 873.000000, 

CUDA_VISIBLE_DEVICES=0 python3 tools/train_zx.py --config_file config/icdar2015_dcn_resnet18_FPN_DBhead_polyLR.yaml
tensorboard --logdir output/DBNet_deformable_resnet18_FPN_DBHead 

# 2021-06-12 01:17:50,017 DBNet.pytorch INFO: current best, recall: 0.692494, precision: 0.911409, hmean: 0.787012, train_loss: 0.525073, best_model_epoch: 1185.000000, 

CUDA_VISIBLE_DEVICES=1 python3 tools/train_zx.py --config_file config/icdar2015_resnet50_FPN_DBhead_polyLR.yaml
tensorboard --logdir output/DBNet_resnet50_FPN_DBHead --port 6007

# 2021-06-12 11:58:15,888 DBNet.pytorch INFO: current best, recall: 0.711864, precision: 0.899083, hmean: 0.794595, train_loss: 0.488941, best_model_epoch: 978.000000, 
```

# eval
```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py --model_path 'output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth'

# FPS:36.34431715208202 (0.7322033898305085, 0.8925619834710744, 0.8044692687917216)
```