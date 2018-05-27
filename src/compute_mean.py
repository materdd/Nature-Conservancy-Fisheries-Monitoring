
#!/usr/bin/env python
import argparse
import sys

import numpy as np

import chainer

import cv2
import numpy as np

def compute_mean(dataset, resize_shape=None):
    print('compute mean image')
    sum_image = 0
    N = len(dataset)
    for i, (image, _) in enumerate(dataset):
        if resize_shape is not None:
            if image.shape[1:] != resize_shape:
                image = image.transpose(1,2,0)
                image = cv2.resize(image, (resize_shape))
                #image = np.transpose(image, (2,0,1))
                image = np.asarray(image).transpose(2,0,1)
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
    sys.stderr.write('\n')
    return sum_image / N


def main():
    parser = argparse.ArgumentParser(description='Compute images mean array')
    parser.add_argument('--dataset', default="../data/all.txt",
                        help='Path to training image-label list file')
    parser.add_argument('--root', '-R', default="/media/matterd/0C228E28228E173C/dataset/kaggle/fisheries_monitoring/train/train/",
                        help='Root directory path of image files')
    parser.add_argument('--output', '-o', default='mean.npy',
                        help='path to output mean array')
    args = parser.parse_args()

    print(args)

    dataset = chainer.datasets.LabeledImageDataset(args.dataset, args.root)
    mean = compute_mean(dataset, resize_shape=(1280, 720))
    np.save(args.output, mean)


if __name__ == '__main__':
    main()