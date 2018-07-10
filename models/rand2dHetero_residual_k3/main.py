#!/usr/bin/env python
import argparse
import os,sys
import numpy as np

#from glob import glob

import tensorflow as tf
from model import denoiser
import scipy.io as sio

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--checkpoint_dir',
    dest='ckpt_dir',
    default='./checkpoint',
    help='models are saved here')

parser.add_argument(
    '--epoch',
    dest='epoch',
    type=int,
    default=50,
    help='# of epoch')

parser.add_argument(
    '--lr',
    dest='lr',
    type=float,
    default=0.001,
    help='initial learning rate for adam')

parser.add_argument(
    '--batch_size',
    dest='batch_size',
    type=int,
    default=64, # Note, I use the same batch size when generating input.
    help='# images in batch')

parser.add_argument(
    '--use_gpu',
    dest='use_gpu',
    type=int,
    default=1,
    help='gpu flag, 1 for GPU and 0 for CPU')

parser.add_argument(
    '--phase',
    dest='phase',
    default='train',
    help='train or test')

args = parser.parse_args()


def denoiser_train(denoiser, lr):
    #
    # load noisy and clean data
    #
    print("[*] Loading data...")

    #--------------------------------------------------------------------------
    #  apply log(x+1) to the raw value, and select maxV to normalize
    #--------------------------------------------------------------------------

    noisy_data = np.load('../../model_input/rand2d_hetero/rand2d_noisy_pats_1e+04.npy')

    #clean_data = np.load('../../model_input/rand2d_hetero/rand2d_clean_pats_1e+04.npy')
    clean_data = np.load('../../model_input/rand2d_hetero/rand2d_resid_pats_1e+04.npy')

    print noisy_data.shape , clean_data.shape


    #--- max value ---#
    print "\nprev log(x+1)"

    print "noisy_max \t clean_max \t max"
    max_noisy, max_clean = np.amax(noisy_data), np.amax(clean_data)
    maxV = max(max_noisy, max_clean)
    print("{}\t{}\t{}".format(max_noisy, max_clean, maxV))


    print "noisy_min \t clean_min "
    min_noisy, min_clean = np.amin(noisy_data), np.amin(clean_data)
    print("{}\t{}".format(min_noisy, min_clean))


    #--- apply log(x + 1) ---#
    noisy_data = np.log(noisy_data + 1.)
    clean_data = np.log(clean_data + 1.)


    print "\nafter log(x+1)"
    print "noisy_max \t clean_max \t max"
    max_noisy, max_clean = np.amax(noisy_data), np.amax(clean_data)
    maxV = max(max_noisy, max_clean)
    print max_noisy, max_clean, maxV


    maxV = float(int(maxV) + 3) # add extra 3 for upper bound

    # if maxV <= 25 use 25, else use maxV
    if maxV <= 25.0:
        maxV = 25.0


    print "Using %f for maxV (after log)" % maxV

    print "\nSaving maxV to use in matlab..."
    sio.savemat('maxV.mat', dict(maxV=maxV))

    print  "\nDone!"


    #--- normalize with maxV ---#
    noisy_data = noisy_data / maxV 
    clean_data = clean_data / maxV 


    #--- run training ---#
    denoiser.train(noisy_data,
            clean_data, 
            batch_size=args.batch_size,
            ckpt_dir=args.ckpt_dir,
            epoch=args.epoch, 
            lr=lr)


def main(_):
    # checkpoint dir
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    lr = args.lr * np.ones([args.epoch])

    # reduce learning rate after 60% total epoch
    small_lr_pos = int(args.epoch * 0.6)
    lr[small_lr_pos:] = lr[0] * 0.1

    if args.use_gpu:
        # added to control the gpu memory
        print("Run Tensorflow [GPU]\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)  # init a denoiser class
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                # denoiser_test(model)
                pass
            else:
                print('[!]Unknown phase')
                sys.exit(1)
    else:
        # print("CPU\n")
        print "CPU Not supported yet!"
        sys.exit(1)


if __name__ == '__main__':
    tf.app.run()
