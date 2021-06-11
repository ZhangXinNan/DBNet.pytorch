
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/home/zhangxin/github/DBNet.pytorch/datasets/train/img')
    parser.add_argument('--gt_dir', default='/home/zhangxin/github/DBNet.pytorch/datasets/train/gt')
    parser.add_argument('--output', default='/home/zhangxin/github/DBNet.pytorch/datasets/train.txt')
    return parser.parse_args()


def main(args):
    print(args)

    with open(args.output, 'w') as fo:
        for filename in os.listdir(args.img_dir):
            name, suffix = os.path.splitext(filename)
            if suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            img_path = os.path.join(args.img_dir, filename)
            gt_path = os.path.join(args.gt_dir, "gt_{}.txt".format(name))
            fo.write("{}\t{}\n".format(img_path, gt_path))


if __name__ == '__main__':
    main(get_args())
