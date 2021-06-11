
import os
import argparse
import PIL
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/home/zhangxin/data_public/ICDAR2019-LSVT/train_full_images_0')
    parser.add_argument('--gt_dir', default='/home/zhangxin/github/DBNet.pytorch/datasets/ICDAR2019-LSVT/val/gt')
    return parser.parse_args()


def main(args):
    for filename in os.listdir(args.img_dir):
        name, suffix = os.path.splitext(filename)
        if suffix.lower() not in ['.jpg']:
            print(filename, 'suffix is not jpg')
            continue
        img_path = os.path.join(args.img_dir, filename)
        gt_path = os.path.join(args.gt_dir, name + '.txt')
        if not os.path.isfile(gt_path):
            print(gt_path, 'is not a file')
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_show = img.copy()
        with open(gt_path, 'r') as fi:
            for i, line in enumerate(fi):
                arr = line.strip().split(',')
                if len(arr) < 9:
                    print("len(arr) < 9")
                    continue
                pts = [float(x) for x in arr[:8]],
                tsc = arr[8]
                print(i, filename, pts, tsc)
                pts = np.array(pts, dtype=np.int).reshape((-1, 2))
                cv2.fillConvexPoly(img_show, pts, (0, 255, 0))
        img_show = cv2.addWeighted(img, 0.5, img_show, 0.5, 0)
        cv2.imshow(filename, img_show)
        cv2.waitKey(25000)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(get_args())