#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.
Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).
"""
from __future__ import print_function
import argparse
import random

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

from network import alex
from network import googlenet
from network import googlenetbn
from network import nin

from config import *
from PIL import Image
import numpy as np
import os
import six
import cv2

def _read_image_as_array(path, dtype, resize_shape=None):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=dtype)
        if resize_shape is not None:
            image = cv2.resize(image, resize_shape)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image

def _postprocess_image(image):
    if image.ndim == 2:
        # image is greyscale
        image = image[..., None]
    return image.transpose(2, 0, 1)

class ResizedImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pairs, root="", dtype=np.float32,
                 label_dtype=np.int32, resize_shape=None):
        #_check_pillow_availability()
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._root = root
        self._dtype = dtype
        self._label_dtype = label_dtype
        self._resize_shape = resize_shape

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = _read_image_as_array(full_path, self._dtype, self._resize_shape)

        label = np.array(int_label, dtype=self._label_dtype)
        #print(full_path)
        #print(image.shape)
        #print(label)
        return _postprocess_image(image), label

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True, resize_shape=None):
        #self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.base = ResizedImageDataset(path, root, resize_shape=resize_shape)
        self.mean = mean
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value

        
        crop_size = self.crop_size
        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image /= 255
        return image, label


def main():
    archs = {
        'alex': alex.Alex,
        'alex_fp16': alex.AlexFp16,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        'nin': nin.NIN
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument("--train", default="../data/train.txt", help='Path to training image-label list file')
    parser.add_argument('--val', default="../data/val.txt", help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='alex',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=16,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', default=8, type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='../mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='../result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default="/media/matterd/0C228E28228E173C/dataset/kaggle/fisheries_monitoring/train/train/",
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=100,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
  
    # Initialize the model to train
    print("----- LOAD MODEL -----")
    model = archs[args.arch]() # ネットワーク構造を定義
    if args.initmodel: # 学習済みモデルがある場合
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model) # 学習済みモデルがを読み込み
    if args.gpu >= 0: # gpuを使う場合
        chainer.cuda.get_device(args.gpu).use()  # 計算に使うgpuを指定
        model.to_gpu() # 学習にgpuを使う

    # Load the datasets and mean file
    print("----- DATASET -----")
    resize_shape = (250,250)
    mean = np.load(args.mean).transpose(1,2,0) # 平均画像を読み込み、(height, width, chanel)に並び替え(opencvの画像フォーマット)
    mean = cv2.resize(mean, (resize_shape)) # 平均画像をリサイズ
    mean = np.array(mean).transpose(2,0,1) # 平均画像を(channel, height, width)に並び替え(chainerの画像フォーマット)
    
    train = PreprocessedDataset(args.train, args.root, mean, model.insize, resize_shape=resize_shape) # trainデータセットの前処理
    val = PreprocessedDataset(args.val, args.root, mean, model.insize, False, resize_shape=resize_shape) # validデータセットの前処理
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    #train_iter = chainer.iterators.MultiprocessIterator(
    #    train, args.batchsize, n_processes=args.loaderjob)
    #val_iter = chainer.iterators.MultiprocessIterator(
    #    val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # trainデータのイテレータを作成
    train_iter = chainer.iterators.SerialIterator( 
        train, args.batchsize)
    # validデータのイテレータを作成
    val_iter = chainer.iterators.SerialIterator(
        val, args.val_batchsize, repeat=False, shuffle=False)

    # Set up an optimizer
    print("----- OPTIMIZER -----")
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9) # optimizerの設定
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10 if args.test else 100000), 'iteration' # validationするタイミングを設定
    log_interval = (10 if args.test else 1000), 'iteration' # logを出すタイミングの設定

    # Copy the chain with shared parameters to flip 'train' flag only in test
    eval_model = model.copy()
    eval_model.train = False

    # その他拡張モジュールの設定
    trainer.extend(extensions.Evaluator(val_iter, eval_model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    print("----- START TRAINING -----")
    trainer.run()
    print("----- FINISH TRAINING -----")


if __name__ == '__main__':
    main()