#!/usr/bin/env python

import glob
import argparse
import sys
import os

# load mat in python
import scipy.io as spio
import numpy as np



def update_data(inputs_noisy, inputs_clean, offset, filepaths, test_id, 
        im_w = 100, im_h =100, imgprefix=None, clean_dir=None, clean_prefix=None, target_axis=None):

    if imgprefix is None:
        print "imgprefix is none! error!"
        sys.exit(1)

    if clean_dir is None:
        print "clean dir is none! error!"
        sys.exit(1)

    if clean_prefix is None:
        print "clean prefix is none! error!"
        sys.exit(1)

    if target_axis not in ['x','y','z']:
        print "wrong axis dir (x,y,z are allowed)"
        sys.exit(1)


    for fid, noisyfile in enumerate(filepaths):
        noisymat = spio.loadmat(noisyfile, squeeze_me=True)
        noisyData = noisymat['currentImage']

        #-------------------------------------------
        # find out the imageID
        # remove the prefix, then the suffix ".mat"
        #-------------------------------------------
        img_prefix_len = len(imgprefix)
        img_id = int(noisyfile[img_prefix_len:][:-4])

        cleanfile = clean_dir + '/' + str(test_id) + '/' + target_axis + '/' + clean_prefix + str(test_id) + '_img'  + str(img_id) + '.mat'

        #print noisyfile
        #print cleanfile
        #break

        cleanmat = spio.loadmat(cleanfile, squeeze_me=True)
        cleanData = cleanmat['currentImage']

        if noisyData.shape[0] != cleanData.shape[0] or noisyData.shape[1] != cleanData.shape[1]:
            print('Error! Noisy data size is different from clean data size!')
            sys.exit(1)

        # rotation 
        # data / data90 / data180/ data270
        noisyData_r90 = np.rot90(noisyData, k=1)
        cleanData_r90 = np.rot90(cleanData, k=1)

        noisyData_r180 = np.rot90(noisyData, k=2)
        cleanData_r180 = np.rot90(cleanData, k=2)

        noisyData_r270 = np.rot90(noisyData, k=3)
        cleanData_r270 = np.rot90(cleanData, k=3)

        # extend one dimension
        noisyData = np.reshape(noisyData, (im_w, im_h, 1))
        cleanData = np.reshape(cleanData, (im_w, im_h, 1))

        # print noisyData.shape

        noisyData_r90 = np.reshape(noisyData_r90, (im_h, im_w, 1))
        cleanData_r90 = np.reshape(cleanData_r90, (im_h, im_w, 1))

        noisyData_r180 = np.reshape(noisyData_r180, (im_h, im_w, 1))
        cleanData_r180 = np.reshape(cleanData_r180, (im_h, im_w, 1))

        noisyData_r270 = np.reshape(noisyData_r270, (im_h, im_w, 1))
        cleanData_r270 = np.reshape(cleanData_r270, (im_h, im_w, 1))

        
        #
        # update noisy and clean array
        #
        inputs_noisy[offset + fid * 4, :, :, :] = noisyData[:, :, :]
        inputs_clean[offset + fid * 4, :, :, :] = cleanData[:, :, :]

        inputs_noisy[offset + fid * 4 + 1, :, :, :] = noisyData_r90[:, :, :]
        inputs_clean[offset + fid * 4 + 1, :, :, :] = cleanData_r90[:, :, :]

        inputs_noisy[offset + fid * 4 + 2, :, :, :] = noisyData_r180[:, :, :]
        inputs_clean[offset + fid * 4 + 2, :, :, :] = cleanData_r180[:, :, :]

        inputs_noisy[offset + fid * 4 + 3, :, :, :] = noisyData_r270[:, :, :]
        inputs_clean[offset + fid * 4 + 3, :, :, :] = cleanData_r270[:, :, :]



#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def gen_osa_data(photon_vol, batch_size = 64, im_w = 100, im_h = 100,
                 test_num=10,
                 data_dir='../data/osa',
                 ground_truth='../data/osa/1e+09',
                 ground_truth_prefix='osa_phn1e+09_test',
                 save_dir = './'):
    '''
    Each simulation results in a different 3D voxel image  using different random seed.
    For osa data sets, we used 100x100x100 3D voxel.
    '''
    target_dir = data_dir + '/' + photon_vol  # locate the simulation folder




    #--------------------------------------------------------------------------
    # count the number of patches
    # Notice: here we use full image
    #--------------------------------------------------------------------------
    img_count = 0
    for test_id in xrange(1, test_num + 1):
        dir_x = target_dir + '/' + str(test_id) + '/x/*.mat'
        dir_y = target_dir + '/' + str(test_id) + '/y/*.mat'
        dir_z = target_dir + '/' + str(test_id) + '/z/*.mat'

        filepaths_dirx = glob.glob(dir_x)    # form the file path
        filepaths_diry = glob.glob(dir_y)
        filepaths_dirz = glob.glob(dir_z)

        img_count += len(filepaths_dirx) + len(filepaths_diry) + len(filepaths_dirz)

    print "[LOG] There are %d images." % img_count

    ##--------------------------------------------------------------------------
    ## 
    ## 
    ##--------------------------------------------------------------------------

    #print "We will rotate them to 90/180/270 degrees. Therefore, we will have 4x images."
    #img_count = img_count * 4
    #print "New total = %d" % img_count

    #if img_count % batch_size != 0:  # if can't be evenly dived by batch size
    #    numPatches = (img_count / batch_size + 1) * batch_size 
    #else:
    #    numPatches = img_count

    #print "[LOG] total patches = %d , batch size = %d, total batches = %d" % \
    #      (numPatches, batch_size, numPatches / batch_size)



    ##--------------------------------------------
    ## data matrix 4-D : model input and output 
    ##--------------------------------------------
    #inputs_noisy = np.zeros((numPatches, im_w, im_h, 1), dtype=np.float32) 
    #inputs_clean = np.zeros((numPatches, im_w, im_h, 1), dtype=np.float32)


    ##--------------------------------------------------------------------------
    ## generate the patches
    ##--------------------------------------------------------------------------
    #offset = 0
    #imgNum = 0

    #for test_id in xrange(1, test_num + 1):

    #    files_in_dirx = target_dir + '/' + str(test_id) + '/x/*.mat'
    #    files_in_diry = target_dir + '/' + str(test_id) + '/y/*.mat'
    #    files_in_dirz = target_dir + '/' + str(test_id) + '/z/*.mat'

    #    filepaths_x = glob.glob(files_in_dirx)
    #    filepaths_y = glob.glob(files_in_diry)
    #    filepaths_z = glob.glob(files_in_dirz)

    #    imgprefix_x = target_dir + '/' + str(test_id) + '/x/osa_phn' + photon_vol + '_test' + str(test_id) + '_img'
    #    imgprefix_y = target_dir + '/' + str(test_id) + '/y/osa_phn' + photon_vol + '_test' + str(test_id) + '_img'
    #    imgprefix_z = target_dir + '/' + str(test_id) + '/z/osa_phn' + photon_vol + '_test' + str(test_id) + '_img'

    #    ##print imgprefix_x
    #    ##print imgprefix_y
    #    ##print imgprefix_z

    #    #----------------------------------------------------------------------
    #    # files for x axis images
    #    #----------------------------------------------------------------------
    #    imgNum = len(filepaths_x)

    #    # for instance, ( 10 test * 100 imgs/test * 4 vers /  img  )*  3 axis
    #    offset = (test_id - 1) * 3 * imgNum * 4  # test_id start from 1, but we need it to start from 0

    #    update_data(inputs_noisy, inputs_clean, offset, filepaths_x, test_id, im_w = im_w, im_h = im_h, 
    #            imgprefix =imgprefix_x, clean_dir = ground_truth, clean_prefix = ground_truth_prefix, target_axis='x')

    #    #----------------------------------------------------------------------
    #    # files for y axis images
    #    #----------------------------------------------------------------------
    #    offset += imgNum * 4

    #    # 4 image along y axis
    #    update_data(inputs_noisy, inputs_clean, offset, filepaths_y, test_id, im_w = im_w, im_h = im_h,
    #            imgprefix =imgprefix_y, clean_dir = ground_truth, clean_prefix = ground_truth_prefix, target_axis='y')

    #    #----------------------------------------------------------------------
    #    # files for z axis images
    #    #----------------------------------------------------------------------

    #    offset += imgNum * 4
    #    # 4 image along z axis
    #    update_data(inputs_noisy, inputs_clean, offset, filepaths_z, test_id, im_w = im_w, im_h = im_h,
    #            imgprefix =imgprefix_z, clean_dir = ground_truth, clean_prefix = ground_truth_prefix, target_axis='z')



    #patch_count = offset + imgNum * 4
    #print '[LOG] %d images are generated!' % (patch_count)


    ##print inputs_noisy[10, 50, 50, :],  inputs_clean[10, 50, 50, :]
    ##print inputs_noisy[100, 50, 50, :], inputs_clean[100, 50, 50, :]


    ##--------------------------------------------------------------------------
    ## pad the array 
    ##--------------------------------------------------------------------------
    #if patch_count < numPatches:
    #    print '[LOG] padding the batch ... '
    #    to_pad = numPatches - patch_count
    #    inputs_noisy[-to_pad:, :, :, :] = inputs_noisy[:to_pad, :, :, :]
    #    inputs_clean[-to_pad:, :, :, :] = inputs_clean[:to_pad, :, :, :]


    ##--------------------------------------------------------------------------
    ## save it to a file 
    ##--------------------------------------------------------------------------
    #print '[LOG] saving data to disk ... '
    #if not os.path.exists(save_dir):
    #    os.mkdir(save_dir)

    #np.save(os.path.join(save_dir, "osa_img_noisy_pats_" + photon_vol), inputs_noisy)

    #np.save(os.path.join(save_dir,"osa_img_clean_pats_" + photon_vol), inputs_clean)

    #print '[LOG] Done! '
    #print '[LOG] Check %s for the output.' % save_dir
    #print "[LOG] size of inputs tensor = " + str(inputs_noisy.shape)


if __name__ == '__main__':

    print '\nGenerating osa data with [ different ] src postion!'
    gen_osa_data('1e+05', test_num=20, save_dir='../model_input/')
