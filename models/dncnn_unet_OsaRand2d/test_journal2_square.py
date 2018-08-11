#!/usr/bin/env python
import argparse
import os
import numpy as np
import tensorflow as tf
import scipy.io as spio

import matplotlib.pyplot as plt
import math

import tensorflow as tf

from model import denoiser

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--checkpoint_dir',
    dest='ckpt_dir',
    default='./checkpoint',
    help='models are saved here')
parser.add_argument(
    '--batch_size',
    dest='batch_size',
    type=int,
    default=64,
    help='# images in batch')
parser.add_argument(
    '--use_gpu',
    dest='use_gpu',
    type=int,
    default=1,
    help='gpu flag, 1 for GPU and 0 for CPU')

args = parser.parse_args()


def denoiser_test(denoiser):

    #--------------------------------------------------------------------------
    # apply log(x+1) to the raw value
    #--------------------------------------------------------------------------

    #------------
    # hom square  
    #------------
    p5  = '../../prepare_data/spie2d_customize/journal2_hom/square_1e5.mat'
    p6  = '../../prepare_data/spie2d_customize/journal2_hom/square_1e6.mat'

    matp5 = spio.loadmat(p5, squeeze_me=True)  # the output is a dict
    matp6 = spio.loadmat(p6, squeeze_me=True)  # the output is a dict

    datap5 = matp5['currentImage']
    datap6 = matp6['currentImage']

    (im_h, im_w) = datap5.shape

    datap5 = np.reshape(datap5, (im_h, im_w, 1))
    datap6 = np.reshape(datap6, (im_h, im_w, 1))

    # normalize data
    datap5 = np.log(datap5 + 1.)
    datap6 = np.log(datap6 + 1.)

    # maxV 
    maxV = spio.loadmat('maxV.mat', squeeze_me=True)  # the output is a dict
    maxV = maxV['maxV']
    print maxV

    datap5 = datap5 / maxV
    datap6 = datap6 / maxV

    input_p5 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix
    input_p6 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    ### update
    input_p5[0, :, :, :] = datap5 
    input_p6[0, :, :, :] = datap6 

    denoiser.test(input_p5,  ckpt_dir=args.ckpt_dir,outFile='j2-hom-square-p5.mat')
    denoiser.test(input_p6,  ckpt_dir=args.ckpt_dir,outFile='j2-hom-square-p6.mat')

    #------------
    # hom square02
    #------------
    p5  = '../../prepare_data/spie2d_customize/journal2_hom/square02_1e5.mat'
    p6  = '../../prepare_data/spie2d_customize/journal2_hom/square02_1e6.mat'

    matp5 = spio.loadmat(p5, squeeze_me=True)  # the output is a dict
    matp6 = spio.loadmat(p6, squeeze_me=True)  # the output is a dict

    datap5 = matp5['currentImage']
    datap6 = matp6['currentImage']

    (im_h, im_w) = datap5.shape

    datap5 = np.reshape(datap5, (im_h, im_w, 1))
    datap6 = np.reshape(datap6, (im_h, im_w, 1))

    # normalize data
    datap5 = np.log(datap5 + 1.)
    datap6 = np.log(datap6 + 1.)

    # maxV 
    maxV = spio.loadmat('maxV.mat', squeeze_me=True)  # the output is a dict
    maxV = maxV['maxV']
    print maxV

    datap5 = datap5 / maxV
    datap6 = datap6 / maxV

    input_p5 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix
    input_p6 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    ### update
    input_p5[0, :, :, :] = datap5 
    input_p6[0, :, :, :] = datap6 

    denoiser.test(input_p5,  ckpt_dir=args.ckpt_dir,outFile='j2-hom-square02-p5.mat')
    denoiser.test(input_p6,  ckpt_dir=args.ckpt_dir,outFile='j2-hom-square02-p6.mat')

    #--------------------------------------------------------------------------
    # absorber 
    #--------------------------------------------------------------------------
    p5  = '../../prepare_data/spie2d_customize/journal2_absorber/square_1e5.mat'
    p6  = '../../prepare_data/spie2d_customize/journal2_absorber/square_1e6.mat'

    matp5 = spio.loadmat(p5, squeeze_me=True)  # the output is a dict
    matp6 = spio.loadmat(p6, squeeze_me=True)  # the output is a dict

    datap5 = matp5['currentImage']
    datap6 = matp6['currentImage']

    (im_h, im_w) = datap5.shape

    datap5 = np.reshape(datap5, (im_h, im_w, 1))
    datap6 = np.reshape(datap6, (im_h, im_w, 1))

    # normalize data
    datap5 = np.log(datap5 + 1.)
    datap6 = np.log(datap6 + 1.)

    # maxV 
    maxV = spio.loadmat('maxV.mat', squeeze_me=True)  # the output is a dict
    maxV = maxV['maxV']
    print maxV

    datap5 = datap5 / maxV
    datap6 = datap6 / maxV

    input_p5 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix
    input_p6 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    ### update
    input_p5[0, :, :, :] = datap5 
    input_p6[0, :, :, :] = datap6 

    denoiser.test(input_p5,  ckpt_dir=args.ckpt_dir,outFile='j2-absorber-square-p5.mat')
    denoiser.test(input_p6,  ckpt_dir=args.ckpt_dir,outFile='j2-absorber-square-p6.mat')

    #------------
    # absorber square02
    #------------
    p5  = '../../prepare_data/spie2d_customize/journal2_absorber/square02_1e5.mat'
    p6  = '../../prepare_data/spie2d_customize/journal2_absorber/square02_1e6.mat'

    matp5 = spio.loadmat(p5, squeeze_me=True)  # the output is a dict
    matp6 = spio.loadmat(p6, squeeze_me=True)  # the output is a dict

    datap5 = matp5['currentImage']
    datap6 = matp6['currentImage']

    (im_h, im_w) = datap5.shape

    datap5 = np.reshape(datap5, (im_h, im_w, 1))
    datap6 = np.reshape(datap6, (im_h, im_w, 1))

    # normalize data
    datap5 = np.log(datap5 + 1.)
    datap6 = np.log(datap6 + 1.)

    # maxV 
    maxV = spio.loadmat('maxV.mat', squeeze_me=True)  # the output is a dict
    maxV = maxV['maxV']
    print maxV

    datap5 = datap5 / maxV
    datap6 = datap6 / maxV

    input_p5 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix
    input_p6 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    ### update
    input_p5[0, :, :, :] = datap5 
    input_p6[0, :, :, :] = datap6 

    denoiser.test(input_p5,  ckpt_dir=args.ckpt_dir,outFile='j2-absorber-square02-p5.mat')
    denoiser.test(input_p6,  ckpt_dir=args.ckpt_dir,outFile='j2-absorber-square02-p6.mat')


def main(_):
    if args.use_gpu:
        # added to control the gpu memory
        print("Run Tensorflow [GPU]\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            denoiser_test(model)
    else:
        print "CPU Not supported yet!"
        sys.exit(1)


if __name__ == '__main__':
    tf.app.run()
