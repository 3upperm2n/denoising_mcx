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

def get_snr_data(matfile):
    dat_px = spio.loadmat(matfile, squeeze_me=True)
    dat_px = dat_px['data']

    # each data array is 4D tensor (x , y, z, samples)
    samples = dat_px.shape[-1]
    im_x = dat_px.shape[0]
    im_z = dat_px.shape[2]

    snr_array = np.zeros((samples, im_x, im_z), dtype=np.float32)

    for i in xrange(samples):
        img50 = dat_px[:,49,:,i]  # for current example, take the 50 image along the y-axis
        snr_array[i,...] = img50 

    return snr_array


def denoiser_test(denoiser):

    #--------------------------------------------------------------------------
    # apply log(x+1) to the raw value
    #--------------------------------------------------------------------------

    # maxV 
    maxV = spio.loadmat('maxV.mat', squeeze_me=True)  # the output is a dict
    maxV = maxV['maxV']
    print maxV


    #--------------------------------------------------------------------------
    # rand2d mcx for a squre image, using yaoshen's journal2 data 
    #--------------------------------------------------------------------------
    #snr_dir = '/home/users/leiming/files_on_pangu/denoising_mcx/prepare_data/spie2d_customize/test_snr/het_absorber_square_x10_yy/1e+05'
    snr_dir = '../../prepare_data/spie2d_customize/test_snr/het_absorber_square_x10_yy/1e+05'

    im_h, im_w = 100, 100
    N = 100

    input_noisy = np.zeros((N, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    #-------#
    # 1e5
    #-------#

    for i in xrange(1, N+1):
        noisy_file = snr_dir + '/test[X].mat'
        noisy_file = noisy_file.replace('[X]', str(i))

        # read mat
        noisymat  = spio.loadmat(noisy_file, squeeze_me=True)  # the output is a dict
        noisyData = noisymat['currentImage']
        noisyData = np.reshape(noisyData, (im_h, im_w, 1))

        # store the data into 4D tensor, indexing starting from 0
        input_noisy[i-1, :, :, :] = noisyData


    # preprocess 
    input_noisy = np.log(input_noisy + 1.)
    input_noisy = input_noisy / maxV

    denoiser.test(input_noisy, ckpt_dir=args.ckpt_dir, outFile='snr_2d_het_absorber_square_x10_p5nn_yy.mat')


    #--------------------------------------------------------------------------
    # end
    #--------------------------------------------------------------------------


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
