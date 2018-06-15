#!/usr/bin/env python

import sys
import numpy as np

print "merging the datasets (homo + hetero)"

# load npy to numpy 

# homo : around 12K images
osa_noisy = np.load('../osa_img_noisy_pats_1e+05.npy')
osa_clean = np.load('../osa_img_clean_pats_1e+05.npy')

(noisyNum,_,_,_) = osa_noisy.shape
(cleanNum,_,_,_) = osa_clean.shape

if noisyNum <> cleanNum:
    print "Error! the samples in osa_noisy and osa_clean do not match!"
    sys.exit(1)

# hetero: around 8K images
rand2d_noisy = np.load('../rand2d/rand2d_noisy_pats_1e+05.npy')
rand2d_clean = np.load('../rand2d/rand2d_clean_pats_1e+05.npy')

(noisyNum,_,_,_) = rand2d_noisy.shape
(cleanNum,_,_,_) = rand2d_clean.shape

if noisyNum <> cleanNum:
    print "Error! the samples in rand2d_noisy and rand2d_clean do not match!"
    sys.exit(1)

#print osa_noisy.shape
#print rand2d_noisy.shape


#----------
# merge datasets
#----------
all_noisy = np.vstack((osa_noisy, rand2d_noisy))
all_clean = np.vstack((osa_clean, rand2d_clean))

#print all_noisy.shape
#print all_clean.shape

print "saving to disk"
np.save("./osa_rand2d_noisy", all_noisy)
np.save("./osa_rand2d_clean", all_clean)

print "Done!"
