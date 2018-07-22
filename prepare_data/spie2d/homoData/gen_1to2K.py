#!/usr/bin/env python

import glob
import argparse
import sys
import os

# load mat in python
import scipy.io as spio
import numpy as np

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def gen_data(startpos = None, endpos = None, batch_size = 64, im_w = 100, im_h = 100,
                 data_dir  ='../../../data/spie2d/hom',
                 save_dir = None):
    '''
    Each simulation results in a different 2D voxel image  using different random seed.
    '''
    if (not startpos) or ( startpos < 1):
        print("start position should be specified and >=1!\n")
        sys.exit(1)

    if (not endpos) or ( endpos < 1):
        print("end position should be specified and >=1!\n")
        sys.exit(1)


    if not save_dir:
        print("no save dir specified!\n")
        sys.exit(1)

    #
    # there are 5 folders under denoising_mcx/data/spie2d/hom
    #
    photon_vol = ['1e+04', '1e+05', '1e+06', '1e+07', '1e+08'] # str format


    #--------------------------------------------------------------------------
    # count the number of images 
    #--------------------------------------------------------------------------
    for phn in photon_vol:
        target_dir = data_dir + '/' + phn  # locate the simulation folder
        dir_files = target_dir + '/*.mat'
        filepaths_dir = glob.glob(dir_files)    # form the file path
        img_count = len(filepaths_dir)
        print("[LOG] There are {} images under {}.".format(img_count,target_dir))

    #--------------------------------------------------------------------------
    # save the mat to numpy for the target tests
    #--------------------------------------------------------------------------
    print("\n[LOG] Generate data from {} to {}.".format(startpos, endpos))


    N = endpos - startpos + 1
    print("\n[LOG ]Total images = {}".format(N))

    #--------------------------------------------
    # data matrix 4-D : model input and output 
    #--------------------------------------------
    inputs_1e4 = np.zeros((N, im_w, im_h, 1), dtype=np.float32) 
    inputs_1e5 = np.zeros((N, im_w, im_h, 1), dtype=np.float32) 
    inputs_1e6 = np.zeros((N, im_w, im_h, 1), dtype=np.float32) 
    inputs_1e7 = np.zeros((N, im_w, im_h, 1), dtype=np.float32) 
    inputs_1e8 = np.zeros((N, im_w, im_h, 1), dtype=np.float32) 



    for phn in photon_vol:
        print("[LOG] photon = {}".format(phn))
        target_dir = data_dir + '/' + phn  # locate the simulation folder
        idx = 0
        for testID in range(startpos, endpos + 1):
            target_file = target_dir + '/' + 'test' + str(testID) + '.mat' 
            #print target_file
            target = spio.loadmat(target_file, squeeze_me=True)
            target = target['currentImage']
            # extend one dimension
            target = np.reshape(target, (im_w, im_h, 1))

            #print target.shape
        
            #
            # update noisy and clean array
            #
            if phn == '1e+04':
                inputs_1e4[idx, :, :, :] = target[:, :, :]
            if phn == '1e+05':
                inputs_1e5[idx, :, :, :] = target[:, :, :]
            if phn == '1e+06':
                inputs_1e6[idx, :, :, :] = target[:, :, :]
            if phn == '1e+07':
                inputs_1e7[idx, :, :, :] = target[:, :, :]
            if phn == '1e+08':
                inputs_1e8[idx, :, :, :] = target[:, :, :]

            idx += 1
            #break
        #break



    # print inputs_1e5[50,50,50,0]
    # print inputs_1e6[50,50,50,0]
    # print inputs_1e7[50,50,50,0]
    # print inputs_1e8[50,50,50,0]



    #--------------------------------------------------------------------------
    # save it to a file 
    #--------------------------------------------------------------------------
    print '[LOG] saving data to disk ... '
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) # rather than mkdir

    np.save(os.path.join(save_dir, "1to2K_1e4"), inputs_1e4)
    np.save(os.path.join(save_dir, "1to2K_1e5"), inputs_1e5)
    np.save(os.path.join(save_dir, "1to2K_1e6"), inputs_1e6)
    np.save(os.path.join(save_dir, "1to2K_1e7"), inputs_1e7)
    np.save(os.path.join(save_dir, "1to2K_1e8"), inputs_1e8)

    print '[LOG] Done! '
    print '[LOG] Check %s for the output.' % save_dir


if __name__ == '__main__':

    print '\nGenerating rand2d homo data.'
    gen_data(startpos = 1, endpos = 2000, save_dir='../../../model_input/spie2d/hom')
