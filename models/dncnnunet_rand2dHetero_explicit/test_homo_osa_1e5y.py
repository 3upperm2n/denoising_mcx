#!/usr/bin/env python
import argparse
import os
import numpy as np
import tensorflow as tf
import scipy.io as spio

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

    # maxV 
    maxV = spio.loadmat('maxV.mat', squeeze_me=True)  # the output is a dict
    maxV = maxV['maxV']
    print maxV



    #------------
    # 1e5 osa simulation: 100 images along y-axis
    #------------
    im_h, im_w = 100, 100
    N = 100

    input_noisy1 = np.zeros((N, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    for i in xrange(1, 101):
        noisy1  = '../../data/osa/1e+05/1/y/osa_phn1e+05_test1_img[X].mat'
        noisy1 = noisy1.replace('[X]', str(i))

        # read mat file into np array
        noisymat1 = spio.loadmat(noisy1, squeeze_me=True)  # the output is a dict
        noisyData1 = noisymat1['currentImage']
        noisyData1 = np.reshape(noisyData1, (im_h, im_w, 1))

        # preprocess
        noisyData1 = np.log(noisyData1 + 1.)
        noisyData1  = noisyData1  / maxV

        # store the data into 4D tensor, indexing starting from 0
        input_noisy1[i-1, :, :, :] = noisyData1


    denoiser.test(input_noisy1, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-test1.mat')





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
