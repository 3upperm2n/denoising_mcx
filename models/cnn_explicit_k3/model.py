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



# refer: 
# [1] https://github.com/timctho/unet-tensorflow/blob/master/model/u_net_tf_v2.py


class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim

        # build model
        self.X  = tf.placeholder(tf.float32, [None, 100, 100, self.input_c_dim], name='noisy_image')
        self.Y_ = tf.placeholder(tf.float32, [None, 100, 100, self.input_c_dim], name='clean_image')


        #---------------------------------------------------------------------#
        # start of model
        #---------------------------------------------------------------------#
        # block 1
        self.cov1 = tf.layers.conv2d(self.X, 64, 3, padding='same', name='cov1', activation=tf.nn.relu, use_bias=False)
        ### block 2 - 9
        self.cov2 = tf.layers.conv2d(self.cov1, 64, 3, padding='same', name='cov2', use_bias=False, activation=tf.nn.relu)
        self.cov3 = tf.layers.conv2d(self.cov2, 64, 3, padding='same', name='cov3', use_bias=False, activation=tf.nn.relu)
        self.cov4 = tf.layers.conv2d(self.cov3, 64, 3, padding='same', name='cov4', use_bias=False, activation=tf.nn.relu)
        self.cov5 = tf.layers.conv2d(self.cov4, 64, 3, padding='same', name='cov5', use_bias=False, activation=tf.nn.relu)
        self.cov6 = tf.layers.conv2d(self.cov5, 64, 3, padding='same', name='cov6', use_bias=False, activation=tf.nn.relu)
        self.cov7 = tf.layers.conv2d(self.cov6, 64, 3, padding='same', name='cov7', use_bias=False, activation=tf.nn.relu)
        self.cov8 = tf.layers.conv2d(self.cov7, 64, 3, padding='same', name='cov8', use_bias=False, activation=tf.nn.relu)
        self.cov9 = tf.layers.conv2d(self.cov8, 64, 3, padding='same', name='cov9', use_bias=False, activation=tf.nn.relu)
        self.cov10 = tf.layers.conv2d(self.cov9, 1, 3, padding='same', name='cov10',use_bias=False)
	self.Y = self.cov10



        #---------------------------------------------------------------------#
        # end of model
        #---------------------------------------------------------------------#

        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)  # use L2 loss

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
                            self.Y_: batch_clean, self.lr: lr[epoch]})


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
                feed_dict={self.X: noisydata})

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

