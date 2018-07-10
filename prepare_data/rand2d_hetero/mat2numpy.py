#!/usr/bin/env python

import glob
import argparse
import sys
import os

# load mat in python
import scipy.io as spio
import numpy as np
from scipy.signal import medfilt



#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def gen_data(photon_vol, batch_size = 64, im_w = 100, im_h = 100,
                 data_dir='../../data/rand2d_hetero',
                 clean_dir ='../../data/rand2d_hetero/1e+08',
                 clean_prefix='test',
                 save_dir = '../../model_input/rand2d_hetero/'):
    '''
    Each simulation results in a different 2D voxel image  using different random seed.
    '''

    #--------------------------------------------------------------------------
    # count the number of patches
    #--------------------------------------------------------------------------
    target_dir = data_dir + '/' + photon_vol  # locate the simulation folder
    dir_files = target_dir + '/*.mat'
    filepaths_dir = glob.glob(dir_files)    # form the file path
    img_count = len(filepaths_dir)

    print "[LOG] There are %d images." % img_count

    #--------------------------------------------------------------------------
    # 
    # 
    #--------------------------------------------------------------------------

    print "[LOG] We will rotate them to 90/180/270 degrees. Therefore, we will have 4x images."
    img_count = img_count * 4

    if img_count % batch_size != 0:  # if can't be evenly dived by batch size
        numPatches = (img_count / batch_size + 1) * batch_size 
    else:
        numPatches = img_count

    print "[LOG] total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, batch_size, numPatches / batch_size)



    #--------------------------------------------
    # data matrix 4-D : model input and output 
    # 1) noisy 2) clean 3) noise_filter
    #--------------------------------------------
    inputs_noisy = np.zeros((numPatches, im_w, im_h, 1), dtype=np.float32) 
    inputs_clean = np.zeros((numPatches, im_w, im_h, 1), dtype=np.float32)
    noise_filter = np.zeros((numPatches, im_w, im_h, 1), dtype=np.float32)


    #--------------------------------------------------------------------------
    # generate the patches
    #--------------------------------------------------------------------------
    offset = 0
    imgNum = 0


    files_in_dir = target_dir + '/*.mat'
    filepaths_ = glob.glob(files_in_dir)
    imgprefix_ = target_dir + '/' +  'test'

    #print imgprefix_

    #----------------------------------------------------------------------
    # files for images
    #----------------------------------------------------------------------
    imgNum = len(filepaths_)


    for fid, noisyfile in enumerate(filepaths_):
        noisymat = spio.loadmat(noisyfile, squeeze_me=True)
        noisyData = noisymat['currentImage']

        #-------------------------------------------
        # find out the imageID
        # remove the prefix, then the suffix ".mat"
        #-------------------------------------------
        img_prefix_len = len(imgprefix_)
        img_id = int(noisyfile[img_prefix_len:][:-4])

        cleanfile = clean_dir + '/' + clean_prefix + str(img_id) + '.mat'

        #print noisyfile
        #print cleanfile
        #break


        cleanmat = spio.loadmat(cleanfile, squeeze_me=True)
        cleanData = cleanmat['currentImage']

        if noisyData.shape[0] != cleanData.shape[0] or noisyData.shape[1] != cleanData.shape[1]:
            print('Error! Noisy data size is different from clean data size!')
            sys.exit(1)


        #residual = cleanData - noisyData
        residual = abs(noisyData - cleanData)  # there could be neg if not using abs()

        #print noisyData[0:10,0]
        #print cleanData[0:10,0]
        #print residual[0:10,0]

        #print residual.shape

        residualF = medfilt(residual, kernel_size = 3) # default kernel size 3
        #residualF = residual



        # rotation 
        # data / data90 / data180/ data270
        noisyData_r90 = np.rot90(noisyData, k=1)
        cleanData_r90 = np.rot90(cleanData, k=1)
        residualF_r90 = np.rot90(residualF, k=1)

        noisyData_r180 = np.rot90(noisyData, k=2)
        cleanData_r180 = np.rot90(cleanData, k=2)
        residualF_r180 = np.rot90(residualF, k=2)

        noisyData_r270 = np.rot90(noisyData, k=3)
        cleanData_r270 = np.rot90(cleanData, k=3)
        residualF_r270 = np.rot90(residualF, k=3)

        # extend one dimension
        noisyData = np.reshape(noisyData, (im_w, im_h, 1))
        cleanData = np.reshape(cleanData, (im_w, im_h, 1))
        residualF = np.reshape(residualF, (im_w, im_h, 1))

        ## print noisyData.shape

        noisyData_r90 = np.reshape(noisyData_r90, (im_h, im_w, 1))
        cleanData_r90 = np.reshape(cleanData_r90, (im_h, im_w, 1))
        residualF_r90 = np.reshape(residualF_r90, (im_h, im_w, 1))

        noisyData_r180 = np.reshape(noisyData_r180, (im_h, im_w, 1))
        cleanData_r180 = np.reshape(cleanData_r180, (im_h, im_w, 1))
        residualF_r180 = np.reshape(residualF_r180, (im_h, im_w, 1))

        noisyData_r270 = np.reshape(noisyData_r270, (im_h, im_w, 1))
        cleanData_r270 = np.reshape(cleanData_r270, (im_h, im_w, 1))
        residualF_r270 = np.reshape(residualF_r270, (im_h, im_w, 1))


        
        #
        # update noisy and clean array
        #
        inputs_noisy[fid * 4, :, :, :] = noisyData[:, :, :]
        inputs_clean[fid * 4, :, :, :] = cleanData[:, :, :]
        noise_filter[fid * 4, :, :, :] = residualF[:, :, :]

        inputs_noisy[fid * 4 + 1, :, :, :] = noisyData_r90[:, :, :]
        inputs_clean[fid * 4 + 1, :, :, :] = cleanData_r90[:, :, :]
        noise_filter[fid * 4 + 1, :, :, :] = residualF_r90[:, :, :]

        inputs_noisy[fid * 4 + 2, :, :, :] = noisyData_r180[:, :, :]
        inputs_clean[fid * 4 + 2, :, :, :] = cleanData_r180[:, :, :]
        noise_filter[fid * 4 + 2, :, :, :] = residualF_r180[:, :, :]

        inputs_noisy[fid * 4 + 3, :, :, :] = noisyData_r270[:, :, :]
        inputs_clean[fid * 4 + 3, :, :, :] = cleanData_r270[:, :, :]
        noise_filter[fid * 4 + 3, :, :, :] = residualF_r270[:, :, :]

        # break


    patch_count = imgNum * 4
    print '[LOG] %d images are generated!' % (patch_count)


    #print inputs_clean[10, 50, 50, :]
    #print inputs_noisy[10, 50, 50, :]
    #print noise_filter[10, 50, 50, :]


    print "\n(noisy) max \t min "
    print np.amax(inputs_noisy), np.amin(inputs_noisy)


    print "\n(residual) max \t min "
    print np.amax(noise_filter), np.amin(noise_filter)





    #--------------------------------------------------------------------------
    # pad the array 
    #--------------------------------------------------------------------------
    if patch_count < numPatches:
        print '[LOG] padding the batch ... '
        to_pad = numPatches - patch_count
        inputs_noisy[-to_pad:, :, :, :] = inputs_noisy[:to_pad, :, :, :]
        inputs_clean[-to_pad:, :, :, :] = inputs_clean[:to_pad, :, :, :]
        noise_filter[-to_pad:, :, :, :] = noise_filter[:to_pad, :, :, :]


    #--------------------------------------------------------------------------
    # save it to a file 
    #--------------------------------------------------------------------------
    print '[LOG] saving data to disk ... '
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.save(os.path.join(save_dir, "rand2d_noisy_pats_" + photon_vol), inputs_noisy)
    np.save(os.path.join(save_dir, "rand2d_clean_pats_" + photon_vol), inputs_clean)
    np.save(os.path.join(save_dir, "rand2d_resid_pats_" + photon_vol), noise_filter)

    print '[LOG] Done! '
    print '[LOG] Check %s for the output.' % save_dir
    print "[LOG] size of inputs tensor = " + str(inputs_noisy.shape)


if __name__ == '__main__':

    print '\nGenerating rand2d data.'
    gen_data('1e+04', save_dir='../../model_input/rand2d_hetero/')
