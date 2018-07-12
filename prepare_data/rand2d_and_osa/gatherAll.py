#!/usr/bin/env python
import sys, os
import numpy as np
import scipy.io as spio

im_w, im_h = 100, 100

#
# osa data
#
osa_noisy = np.load('../../model_input/osa_img_noisy_pats_1e+05.npy')
osa_clean = np.load('../../model_input/osa_img_clean_pats_1e+05.npy')
print "[LOG] OSA"
print("noisy : {} \t clean : {}\n".format(osa_noisy.shape, osa_clean.shape))


#
# homo data
#
homo_noisy = np.load('../../model_input/rand2d_homo_partial/rand2d_noisy_pats_1e+04.npy')
homo_clean = np.load('../../model_input/rand2d_homo_partial/rand2d_clean_pats_1e+04.npy')
print "[LOG] rand2d homo"
print("noisy : {} \t clean : {}\n".format(homo_noisy.shape, homo_clean.shape))


#
# hetero_data 
#
hetero_noisy = np.load('../../model_input/rand2d_hetero/rand2d_noisy_pats_1e+04.npy')
hetero_clean = np.load('../../model_input/rand2d_hetero/rand2d_clean_pats_1e+04.npy')
print "[LOG] rand2d hetero"
print("noisy : {} \t clean : {}\n".format(hetero_noisy.shape, hetero_clean.shape))


# for exmaple : (23616, 100, 100, 1)
osa_samples = osa_noisy.shape[0]
hom_samples = homo_noisy.shape[0]
het_samples = hetero_noisy.shape[0]

#print osa_samples
#print hom_samples
#print het_samples

total_samples = osa_samples + hom_samples + het_samples 
print("[LOG] Total images = {}".format(total_samples))

inputs_noisy = np.zeros((total_samples, im_w, im_h, 1), dtype=np.float32) 
inputs_clean = np.zeros((total_samples, im_w, im_h, 1), dtype=np.float32) 


# 1) read osa
offset = 0
for ii in xrange(osa_samples):
    inputs_noisy[offset + ii,...] = osa_noisy[ii,...]
    inputs_clean[offset + ii,...] = osa_clean[ii,...]


# 2) read homo 
offset = osa_samples
for ii in xrange(hom_samples):
    try:
        inputs_noisy[offset + ii,...] = homo_noisy[ii,...]
        inputs_clean[offset + ii,...] = homo_clean[ii,...]
    except:
        print ii, offset + ii
        print "[Error] Something is wrong!\n"
        break
    #break


# 3) read hetero
offset = osa_samples + hom_samples
for ii in xrange(het_samples):
    inputs_noisy[offset + ii,...] = hetero_noisy[ii,...]
    inputs_clean[offset + ii,...] = hetero_clean[ii,...]

print("noisy {} \t clean : {}\n".format(inputs_noisy.shape, inputs_clean.shape))

#print total_samples / 64.


#
# save to output
#

#--------------------------------------------------------------------------
# save it to a file 
#--------------------------------------------------------------------------
print '[LOG] saving data to disk ... '

save_dir = '../../model_input/osa_homo_hetero'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

np.save(os.path.join(save_dir, "noisy_pats"), inputs_noisy)
np.save(os.path.join(save_dir, "clean_pats"), inputs_clean)

print '[LOG] Done! '
print '[LOG] Check %s for the output.' % save_dir
