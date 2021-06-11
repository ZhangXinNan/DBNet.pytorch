
import os
import argparse
import json
import random
import numpy as np



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir_list', nargs="+",
                        default=['/home/zhangxin/data_public/ICDAR2019-LSVT/train_full_images_0',
                                 '/home/zhangxin/data_public/ICDAR2019-LSVT/train_full_images_1'])
    parser.add_argument('--label_file', default='/home/zhangxin/data_public/ICDAR2019-LSVT/train_full_labels.json')
    parser.add_argument('--train_dir', default='/home/zhangxin/github/DBNet.pytorch/datasets/ICDAR2019-LSVT/train')
    parser.add_argument('--val_dir', default='/home/zhangxin/github/DBNet.pytorch/datasets/ICDAR2019-LSVT/val')
    parser.add_argument('--train_list', default='train.txt')
    parser.add_argument('--val_list', default='val.txt')
    parser.add_argument('--val_num', default=3000, type=int)
    return parser.parse_args()


def main(args):
    train_gt_dir = os.path.join(args.train_dir, 'gt')
    val_gt_dir = os.path.join(args.val_dir, 'gt')
    if not os.path.isdir(train_gt_dir):
        os.makedirs(train_gt_dir)
    if not os.path.isdir(val_gt_dir):
        os.makedirs(val_gt_dir)

    with open(args.label_file, 'r') as fi:
        annot = json.loads(fi.read())
        name_list = list(annot.keys())
        random.shuffle(name_list)
    print(len(name_list))

    img_path_map = {}
    for img_dir in args.img_dir_list:
        for filename in os.listdir(img_dir):
            name, suffix = os.path.splitext(filename)
            if suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue
            img_path = os.path.join(img_dir, filename)
            img_path_map[name] = img_path
    print(len(img_path_map))

    fo_train = open(args.train_list, 'w')
    fo_val = open(args.val_list, 'w')
    for i, name in enumerate(name_list):
        if name not in img_path_map:
            continue
        img_path = img_path_map[name]

        info_list = []
        for item in annot[name]:
            points = item['points']
            if len(points) != 4:
                print(img_path, item)
                continue
            points = np.array(points, dtype=np.float32).reshape((8)).tolist()
            info_list.append("{},{}\n".format(','.join([str(x) for x in points]), item['transcription']))

        if len(info_list) < 1:
            continue
        if i < args.val_num:
            gt_file = os.path.join(val_gt_dir, name + '.txt')
            fo_val.write("{}\t{}\n".format(img_path, gt_file))
        else:
            gt_file = os.path.join(train_gt_dir, name + '.txt')
            fo_train.write("{}\t{}\n".format(img_path, gt_file))

        with open(gt_file, 'w') as fo:
            for info in info_list:
                fo.write(info)

    fo_train.close()
    fo_val.close()


if __name__ == '__main__':
    main(get_args())
