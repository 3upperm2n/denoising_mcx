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

    # # maxV 
    # maxV = spio.loadmat('maxV.mat', squeeze_me=True)  # the output is a dict
    # maxV = maxV['maxV']
    # print maxV


    #--------------------------------------------------------------------------
    # read yaoshen's journal2 data 
    #--------------------------------------------------------------------------
    snr_dir = '/space/neza/2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data'

    #-------#
    # 1e5
    #-------#
    snr_dat = get_snr_data(snr_dir + '/journal2_50ns_vol100_refractive_1e5_pack1.mat')  # return 100 x (100x100)
    print("[Test] input shape : {}, min_val = {}\n".format(snr_dat.shape, np.amin(snr_dat)))
    [samples, im_h, im_w] = snr_dat.shape
    input_dat = np.reshape(snr_dat, (samples, im_h, im_w, 1)) # extend 1 dimension

    # preprocess
    input_dat = np.log(input_dat + 1.)
    # input_dat = input_dat / maxV

    denoiser.test(input_dat, ckpt_dir=args.ckpt_dir, outFile='snr_refractive1e5_nn.mat')

    ##-------#
    ## 1e6
    ##-------#
    #snr_dat = get_snr_data(snr_dir + '/journal2_50ns_vol100_homo_1e6_pack1.mat')  # return 100 x (100x100)
    #print("[Test] input shape : {}, min_val = {}\n".format(snr_dat.shape, np.amin(snr_dat)))
    #[samples, im_h, im_w] = snr_dat.shape
    #input_dat = np.reshape(snr_dat, (samples, im_h, im_w, 1)) # extend 1 dimension

    ## preprocess
    #input_dat = np.log(input_dat + 1.)
    #input_dat = input_dat / maxV

    #denoiser.test(input_dat, ckpt_dir=args.ckpt_dir, outFile='snr_hom1e6_nn.mat')

    ##-------#
    ## 1e7
    ##-------#
    #snr_dat = get_snr_data(snr_dir + '/journal2_50ns_vol100_homo_1e7_pack1.mat')  # return 100 x (100x100)
    #print("[Test] input shape : {}, min_val = {}\n".format(snr_dat.shape, np.amin(snr_dat)))
    #[samples, im_h, im_w] = snr_dat.shape
    #input_dat = np.reshape(snr_dat, (samples, im_h, im_w, 1)) # extend 1 dimension

    ## preprocess
    #input_dat = np.log(input_dat + 1.)
    #input_dat = input_dat / maxV

    #denoiser.test(input_dat, ckpt_dir=args.ckpt_dir, outFile='snr_hom1e7_nn.mat')

    ##-------#
    ## 1e8
    ##-------#
    #snr_dat = get_snr_data(snr_dir + '/journal2_50ns_vol100_homo_1e8_pack1.mat')  # return 100 x (100x100)
    #print("[Test] input shape : {}, min_val = {}\n".format(snr_dat.shape, np.amin(snr_dat)))
    #[samples, im_h, im_w] = snr_dat.shape
    #input_dat = np.reshape(snr_dat, (samples, im_h, im_w, 1)) # extend 1 dimension

    ## preprocess
    #input_dat = np.log(input_dat + 1.)
    #input_dat = input_dat / maxV

    #denoiser.test(input_dat, ckpt_dir=args.ckpt_dir, outFile='snr_hom1e8_nn.mat')



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
