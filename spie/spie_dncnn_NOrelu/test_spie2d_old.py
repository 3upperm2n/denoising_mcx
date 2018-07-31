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

def mat2np(matFile, maxV=25.0, im_h=100, im_w=100):
    noisymat1 = spio.loadmat(matFile, squeeze_me=True)  # the output is a dict
    noisyData1 = noisymat1['currentImage']
    noisyData1 = np.reshape(noisyData1, (im_h, im_w, 1))
    # preprocess
    noisyData1 = np.log(noisyData1 + 1.)
    noisyData1  = noisyData1  / maxV
    return noisyData1


def denoiser_test(denoiser):

    #--------------------------------------------------------------------------
    # apply log(x+1) to the raw value
    #--------------------------------------------------------------------------

    # maxV 
    maxV = spio.loadmat('maxV.mat', squeeze_me=True)  # the output is a dict
    maxV = maxV['maxV']
    print maxV


    #
    # train model with 1-2K hom, and 1-4K het 
    # test the model with 2001
    #

    im_h, im_w = 100, 100
    N = 4  #  1e4, 1e5, 1e6, 1e7



    #--------------------------------------------------------------------------
    # het 
    #--------------------------------------------------------------------------

    input_noisy = np.zeros((N, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    input_noisy[0, :, :, :] = mat2np('../../data/spie2d_old/het/1e+04/test4001.mat', maxV=maxV, im_h=im_h, im_w=im_w)
    input_noisy[1, :, :, :] = mat2np('../../data/spie2d_old/het/1e+05/test4001.mat', maxV=maxV, im_h=im_h, im_w=im_w)
    input_noisy[2, :, :, :] = mat2np('../../data/spie2d_old/het/1e+06/test4001.mat', maxV=maxV, im_h=im_h, im_w=im_w)
    input_noisy[3, :, :, :] = mat2np('../../data/spie2d_old/het/1e+07/test4001.mat', maxV=maxV, im_h=im_h, im_w=im_w)

    print("[Test] input_noisy shape : {}, min_val = {}\n".format(input_noisy.shape,
        np.amin(input_noisy)))

    denoiser.test(input_noisy, ckpt_dir=args.ckpt_dir, outFile='test_het.mat')


    #--------------------------------------------------------------------------
    # hom 
    #--------------------------------------------------------------------------

    input_noisy = np.zeros((N, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    input_noisy[0, :, :, :] = mat2np('../../data/spie2d_old/hom/1e+04/test2001.mat', maxV=maxV, im_h=im_h, im_w=im_w)
    input_noisy[1, :, :, :] = mat2np('../../data/spie2d_old/hom/1e+05/test2001.mat', maxV=maxV, im_h=im_h, im_w=im_w)
    input_noisy[2, :, :, :] = mat2np('../../data/spie2d_old/hom/1e+06/test2001.mat', maxV=maxV, im_h=im_h, im_w=im_w)
    input_noisy[3, :, :, :] = mat2np('../../data/spie2d_old/hom/1e+07/test2001.mat', maxV=maxV, im_h=im_h, im_w=im_w)

    print("[Test] input_noisy shape : {}, min_val = {}\n".format(input_noisy.shape,
        np.amin(input_noisy)))

    denoiser.test(input_noisy, ckpt_dir=args.ckpt_dir, outFile='test_hom.mat')


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
