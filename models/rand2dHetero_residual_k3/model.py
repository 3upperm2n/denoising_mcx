import time
import os,sys
import random

import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import scipy.io as sio                                                  

def save_results_all(target, target_outfile):
    import scipy.io as sio                                                  
    h = target.shape[1]
    w = target.shape[2]
    img_num = target.shape[3]

    local_img = np.zeros((img_num,h,w))

    for i in xrange(img_num):
        local_img[i,:,:] = target[0,:,:,i]

    sio.savemat(target_outfile, {'local_img': local_img})



def save_results(target, target_name):
    import scipy.io as sio                                                  
    img_num = target.shape[3]
    for i in xrange(img_num):
        local_img = target[0,:,:,i]
        #print target_outfile
        sio.savemat(target_outfile, {'local_img': local_img})




class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=64):
        self.sess = sess
        self.input_c_dim = input_c_dim

        # build model
        self.X  = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='noisy_image')
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='residual_noise')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        #self.Y = dncnn_unet(self.X, is_training=self.is_training) # trained output


        #---------------------------------------------------------------------#
        # start of model
        #---------------------------------------------------------------------#
        # block 1
        self.cov1 = tf.layers.conv2d(self.X, 64, 3, padding='same', name='cov1', activation=tf.nn.relu)

        # block 2 - 9
        self.cov2 = tf.layers.conv2d(self.cov1, 64, 3, padding='same', name='cov2', use_bias=False)
        self.act2 = tf.nn.relu(tf.layers.batch_normalization(self.cov2, training=self.is_training))

        self.cov3 = tf.layers.conv2d(self.act2, 64, 3, padding='same', name='cov3', use_bias=False)
        self.act3 = tf.nn.relu(tf.layers.batch_normalization(self.cov3, training=self.is_training))

        self.cov4 = tf.layers.conv2d(self.act3, 64, 3, padding='same', name='cov4', use_bias=False)
        self.act4 = tf.nn.relu(tf.layers.batch_normalization(self.cov4, training=self.is_training))

        self.cov5 = tf.layers.conv2d(self.act4, 64, 3, padding='same', name='cov5', use_bias=False)
        self.act5 = tf.nn.relu(tf.layers.batch_normalization(self.cov5, training=self.is_training))

        self.cov6 = tf.layers.conv2d(self.act5, 64, 3, padding='same', name='cov6', use_bias=False)
        self.act6 = tf.nn.relu(tf.layers.batch_normalization(self.cov6, training=self.is_training))

        self.cov7 = tf.layers.conv2d(self.act6, 64, 3, padding='same', name='cov7', use_bias=False)
        self.act7 = tf.nn.relu(tf.layers.batch_normalization(self.cov7, training=self.is_training))

        self.cov8 = tf.layers.conv2d(self.act7, 64, 3, padding='same', name='cov8', use_bias=False)
        self.act8 = tf.nn.relu(tf.layers.batch_normalization(self.cov8, training=self.is_training))

        self.cov9 = tf.layers.conv2d(self.act8, 64, 3, padding='same', name='cov9', use_bias=False)
        self.act9 = tf.nn.relu(tf.layers.batch_normalization(self.cov9, training=self.is_training))

        # block 10
        self.cov10 = tf.layers.conv2d(self.act9, 1, 3, padding='same')


        # deduct the noise from the input
        self.dncnn_out = self.X - self.cov10 


        ##
        ## we will use unet to learn the left noise (difusion pattern)
        ##

        ## conv + conv + max_pool
        #self.down0a = tc.layers.conv2d(self.cov10,  64,  (3,3), padding='same', normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        ##self.down0a = tc.layers.conv2d(self.dncnn_out,  64,  (3,3), padding='same', normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.down0b = tc.layers.conv2d(self.down0a,     64,  (3,3), padding='same', normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.down0c = tc.layers.max_pool2d(self.down0b,      (2,2), padding='same')

        ## down 1
        #self.down1a = tc.layers.conv2d(self.down0c,  128, (3,3), padding='same', normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.down1b = tc.layers.conv2d(self.down1a,  128, (3,3), padding='same', normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.down1c = tc.layers.max_pool2d(self.down1b,   (2,2), padding='same')

        ## down 2
        #self.down2a = tc.layers.conv2d(self.down1c,  256, (3,3), padding='same', normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.down2b = tc.layers.conv2d(self.down2a,  256, (3,3), padding='same', normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})

        ## up 1
        #self.up1a = tc.layers.conv2d_transpose(self.down2b, 128, (2,2), 2, normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.up1b = tf.concat([self.up1a, self.down1b], axis=3)
        #self.up1c = tc.layers.conv2d(self.up1b, 128, (3,3), normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.up1d = tc.layers.conv2d(self.up1c, 128, (3,3), normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.up1e = tc.layers.conv2d(self.up1d, 128, (3,3), normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})

        #self.up0a = tc.layers.conv2d_transpose(self.up1e, 64, (2,2), 2, normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.up0b = tf.concat([self.up0a, self.down0b], axis=3)
        #self.up0c = tc.layers.conv2d(self.up0b, 64, (3,3), normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.up0d = tc.layers.conv2d(self.up0c, 64, (3,3), normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})
        #self.up0e = tc.layers.conv2d(self.up0d, 64, (3,3), normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': self.is_training})

        ## NOTE: can we have multiple output, and select the max val for each position among them ?
        ##self.output_unet = tc.layers.conv2d(self.up0e, 1, [1, 1], activation_fn=None)
        #self.Y = tc.layers.conv2d(self.up0e, 1, [1, 1], activation_fn=None)

        self.Y = self.dncnn_out







        #---------------------------------------------------------------------#
        # end of model
        #---------------------------------------------------------------------#

        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)  # use L2 loss
        #self.loss = (1.0 / batch_size) * tf.reduce_sum(tf.abs(self.Y_ - self.Y))  # use L1 loss



        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        #self.eva_psnr = tf_psnr(self.Y, self.Y_)

        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")


    def load(self, checkpoint_dir):
        '''
        read checkpoint
        '''
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0



    #--------------------------------------------------------------------------
    # training
    #--------------------------------------------------------------------------
    def train(self, noisy_data, clean_data, batch_size, ckpt_dir, epoch, lr, eval_every_epoch=10):

        # assert data range is between 0 and 1
        numBatch = int(noisy_data.shape[0] / batch_size)

        #-----------------------
        # load pretrained model
        #-----------------------
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")


        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()


        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))

        start_time = time.time()

        ##self.evaluate(iter_num, eval_data, sample_dir=sample_dir, 
        ##        summary_merged=summary_psnr,
        ##        summary_writer=writer)  # eval_data value range is 0-255


        samples = noisy_data.shape[0]

        for epoch in xrange(start_epoch, epoch):
            ## randomize data 
            random.seed(a=epoch)
            row_idx = np.arange(samples)
            np.random.shuffle(row_idx)

            noisy_input = noisy_data[row_idx,:]
            clean_input = clean_data[row_idx,:]
            #print noisy_new.shape, clean_new.shape
            

            for batch_id in xrange(start_step, numBatch):
                batch_noisy = noisy_input[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                batch_clean = clean_input[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]


                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                        feed_dict={self.X: batch_noisy, 
                            self.Y_: batch_clean, 
                            self.lr: lr[epoch],
                            self.is_training: True})


                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))

                iter_num += 1
                #writer.add_summary(summary, iter_num)

            if np.mod(epoch + 1, eval_every_epoch) == 0:
                #self.evaluate(iter_num, eval_data, summary_merged=summary_psnr, summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)

        print("[*] Finish training.")


    def save(self, iter_num, ckpt_dir, model_name='unet-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)


    def test(self, noisydata, ckpt_dir, outFile=''):
        """Test unet"""
        import scipy.io as sio

        # init variables
        tf.initialize_all_variables().run()
        assert len(noisydata) != 0, 'No testing data!'

        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")

        ## note: input is 4D
        startT = time.time()

        output_clean_image = self.sess.run([self.Y],
                feed_dict={self.X: noisydata, self.is_training: False})

        endT = time.time()
        print("=> denoiser runtime = {} (s)".format(endT - startT))


        output_clean_image = np.asarray(output_clean_image)
        #print output_clean_image.shape

        output_clean = output_clean_image[0,:,:,:,0]  # (1, 100, 100, 100, 1)
        #print output_clean.shape   # 100 x 100 x 100

        if len(outFile) == 0:
            return output_clean
        else:
            # save output_clean to mat file
            sio.savemat(outFile, {'output_clean':output_clean})




    def plot(self, noisydata, ckpt_dir, outDir='./'):
        """Test unet"""
        import scipy.io as sio

        # init variables
        tf.initialize_all_variables().run()
        assert len(noisydata) != 0, 'No testing data!'

        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")

        ## note: input is 4D
        cov1_imgs, \
        cov2_imgs, act2_imgs, \
        cov3_imgs, act3_imgs, \
        cov4_imgs, act4_imgs, \
        cov5_imgs, act5_imgs, \
        cov6_imgs, act6_imgs, \
        cov7_imgs, act7_imgs, \
        cov8_imgs, act8_imgs, \
        cov9_imgs, act9_imgs, \
        cov10_imgs, \
        dncnnout_imgs, \
        down0a_imgs, down0b_imgs, down0c_imgs, \
        down1a_imgs, down1b_imgs, down1c_imgs, \
        down2a_imgs, down2b_imgs, \
        up1c_imgs, up1d_imgs, up1e_imgs, \
        up0c_imgs, up0d_imgs, up0e_imgs, \
        final_imgs = self.sess.run([self.cov1,
            self.cov2,
            self.act2,
            self.cov3,
            self.act3,
            self.cov4,
            self.act4,
            self.cov5,
            self.act5,
            self.cov6,
            self.act6,
            self.cov7,
            self.act7,
            self.cov8,
            self.act8,
            self.cov9,
            self.act9,
            self.cov10,
            self.dncnn_out,
            #
            self.down0a,
            self.down0b,
            self.down0c,
            # down 1
            self.down1a,
            self.down1b,
            self.down1c,
            # down 2
            self.down2a,
            self.down2b,
            # up 1
            #self.up1a,
            #self.up1b,
            self.up1c,
            self.up1d,
            self.up1e,
            # up 0
            #self.up0a,
            #self.up0b,
            self.up0c,
            self.up0d,
            self.up0e,
            self.Y],
            feed_dict={self.X: noisydata, self.is_training: False})

        #print cov1_imgs.shape
        #print dncnnout_imgs.shape
        #print final_imgs.shape

        #--- outDir --- #
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        # the output results are 4D tensors
        # save the current image to a mat file

        # cov1
        target_name = outDir + '/cov1.mat'
        target      =            cov1_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov2.mat'
        target      =            cov2_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/act2.mat'
        target      =            act2_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov3.mat'
        target      =            cov3_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/act3.mat'
        target      =            act3_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov4.mat'
        target      =            cov4_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/act4.mat'
        target      =            act4_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov5.mat'
        target      =            cov5_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/act5.mat'
        target      =            act5_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov6.mat'
        target      =            cov6_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/act6.mat'
        target      =            act6_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov7.mat'
        target      =            cov7_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/act7.mat'
        target      =            act7_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov8.mat'
        target      =            cov8_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/act8.mat'
        target      =            act8_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov9.mat'
        target      =            cov9_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/act9.mat'
        target      =            act9_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/cov10.mat'
        target      =            cov10_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/dncnnout.mat'
        target      =            dncnnout_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/down0a.mat'
        target      =            down0a_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/down0b.mat'
        target      =            down0b_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/down0c.mat'
        target      =            down0c_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/down1a.mat'
        target      =            down1a_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/down1b.mat'
        target      =            down1b_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/down1c.mat'
        target      =            down1c_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/down2a.mat'
        target      =            down2a_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/down2b.mat'
        target      =            down2b_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/up1c.mat'
        target      =            up1c_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/up1d.mat'
        target      =            up1d_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/up1e.mat'
        target      =            up1e_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/up0c.mat'
        target      =            up0c_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/up0d.mat'
        target      =            up0d_imgs
        save_results_all(target, target_name)
        target_name = outDir + '/up0e.mat'
        target      =            up0e_imgs
        save_results_all(target, target_name)

        target_name = outDir + '/final.mat'
        target      =            final_imgs
        save_results_all(target, target_name)




    # filter on a jpg image
    def test_fun(self, blue_new, green_new, red_new, ckpt_dir):
        tf.initialize_all_variables().run()

        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")

        ## note: input is 4D
        out_blue = self.sess.run([self.Y], feed_dict={self.X: blue_new, self.is_training: False})
        out_green = self.sess.run([self.Y], feed_dict={self.X: green_new, self.is_training: False})
        out_red = self.sess.run([self.Y], feed_dict={self.X: red_new, self.is_training: False})

        out_blue = np.asarray(out_blue)
        out_green = np.asarray(out_green)
        out_red = np.asarray(out_red)

        print out_blue.shape

