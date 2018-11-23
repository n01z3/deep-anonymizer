from __future__ import print_function
import os
import cv2
import tensorflow as tf
from PIL import Image
import scipy.io as sio
import utils

import contextlib
import timeit
import tqdm 

@contextlib.contextmanager
def elapsed_timer():
    start = timeit.default_timer()
    elapser = lambda: timeit.default_timer() - start
    yield lambda: elapser()
    end = timeit.default_timer()
    elapser = lambda: end - start


def print_time(timer, massage, flg_show=True):
    if flg_show:
        print(
            massage + "{}".format(timer())
        )


class PGN():

    def __init__(self):
        # Create queue coordinator.
        self.graph_pgn = tf.Graph()
        self.coord = tf.train.Coordinator()

    def load_data(self
                , path_data='./datasets/CIHP', path_list_image='./datasets/CIHP/list/val.txt', path_list_image_id=None
                , size_image=(256,256)
    ):

        self.path_data = path_data
        self.path_list_image = path_list_image
        self.path_list_image_id = path_list_image_id

        # set number of prediction iterations
        with open(self.path_list_image, 'r') as f:
            self.n_step = len(f.readlines()) 

        # graph definition
        with self.graph_pgn.as_default() as g :
            # Load reader.
            with tf.name_scope("create_inputs"):
                reader = utils.ImageReader(
                    self.path_data, self.path_list_image, self.path_list_image_id
                    , size_image
                    , False, False, False, self.coord
                )
                # image, label, edge_gt = reader.image, reader.label, reader.edge
                self.image = reader.image
                self.image_rev = tf.reverse(self.image, tf.stack([1]))
                self.image_list = reader.image_list

    def _build_model(self, tta):

        if 1.0 in tta: 
            tta.remove(1.0)
            
        tta.insert(0, 1.0)
        print(tta)  
        # graph definition
        with self.graph_pgn.as_default() as g :
                
            image_batch = tf.stack([self.image, self.image_rev])
            # label_batch = tf.expand_dims(label, dim=0) # Add one batch dimension.
            # edge_gt_batch = tf.expand_dims(edge_gt, dim=0)
            h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])

        
            parsing_out1_list = []
            parsing_out2_list = []
            edge_out2_list = []

            for scale in tta:
                
                if scale != 1.0:
                    tb = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, scale)), 
                                                                       tf.to_int32(tf.multiply(w_orig, scale))]))
                else:
                    tb = image_batch

                with tf.variable_scope('', reuse=False if scale == 1.0 else True):
                    net = utils.model_pgn.PGNModel({'data': tb}, is_training=False, n_classes=self.n_class)

                out1 = tf.image.resize_images(net.layers['parsing_fc'], tf.shape(image_batch)[1:3,])
                out2 = tf.image.resize_images(net.layers['parsing_rf_fc'], tf.shape(image_batch)[1:3,])

                parsing_out1_list.append(out1)
                parsing_out2_list.append(out2)

                if scale >= 1.0:
                    edge_out2 = tf.image.resize_images(net.layers['edge_rf_fc'], tf.shape(image_batch)[1:3,])
                    edge_out2_list.append(edge_out2)


            # --------------
            #  Do something with semantic segmentation
            # --------------

            raw_output = tf.reduce_mean(tf.stack(parsing_out1_list + parsing_out2_list), axis=0)
            head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
            tail_list = tf.unstack(tail_output, num=20, axis=2)
            tail_list_rev = [None] * 20
            for xx in range(14):
                tail_list_rev[xx] = tail_list[xx]
            tail_list_rev[14] = tail_list[15]
            tail_list_rev[15] = tail_list[14]
            tail_list_rev[16] = tail_list[17]
            tail_list_rev[17] = tail_list[16]
            tail_list_rev[18] = tail_list[19]
            tail_list_rev[19] = tail_list[18]
            tail_output_rev = tf.stack(tail_list_rev, axis=2)
            tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
            
            raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
            raw_output_all = tf.expand_dims(raw_output_all, dim=0)
            self.pred_scores = tf.reduce_max(raw_output_all, axis=3)
            raw_output_all = tf.argmax(raw_output_all, axis=3)
            self.pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.

            # --------------
            #  Do something with edges
            # --------------
            raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0) # ???????????????????
            head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
            tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
            raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
            raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
            self.pred_edge = tf.sigmoid(raw_edge_all)
            # res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)


            edge_out2 = tf.reduce_mean(tf.stack(edge_out2_list), axis=0)                     


    def build_model(self, n_class=20, path_model_trained='./checkpoint/CIHP_pgn', tta=[0.50, 0.75, 1.25, 1.50, 1.75]):
        self.n_class = n_class
        self.path_model_trained = path_model_trained

        # define PGN graph
        self._build_model(tta)

        # intialize & load trained model
        with self.graph_pgn.as_default() as g :
            # Which variables to load.
            restore_var = tf.global_variables()
            # Set up tf session and initialize variables. 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            
            self.sess.run(init)
            self.sess.run(tf.local_variables_initializer())
            
            # Load weights.
            loader = tf.train.Saver(var_list=restore_var)
            if self.path_model_trained is not None:
                if utils.utils.load(loader, self.sess, self.path_model_trained):
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")

    def predict(self, flg_debug=False):

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        # evaluate prosessing
        # parsing_dir = './output/cihp_parsing_maps'
        
        # edge_dir = './output/cihp_edge_maps'
        

        # Iterate over training steps.
        with elapsed_timer() as timer_batch:
            print_time(timer_batch, "start iteration\n\t", flg_show=flg_debug)

            for step in tqdm.tqdm(range(self.n_step)):
                with elapsed_timer() as timer_step:

                    # prediction
                    print_time(timer_step, "step={} start\n\t".format(step), flg_show=flg_debug)
                    parsing_, scores, edge_ = self.sess.run([self.pred_all, self.pred_scores, self.pred_edge])

                    print_time(timer_step, "\t done prediction \n\t\t", flg_show=flg_debug)

                    dirname  = os.path.dirname(self.image_list[step])
                    basename = os.path.basename(self.image_list[step])

                    outdir = dirname + '/cihp'

                    
                    edge_dir = outdir + '/edge_maps'
                    if not os.path.exists(edge_dir):
                        os.makedirs(edge_dir)

                    parsing_dir = outdir + '/parsing_maps'
                    if not os.path.exists(parsing_dir):
                        os.makedirs(parsing_dir)
            
                    img_id = '.'.join(basename.split('.')[:-1])

                    cv2.imwrite('{}/{}.png'.format(parsing_dir, img_id), parsing_[0,:,:,0])
                    sio.savemat('{}/{}.mat'.format(parsing_dir, img_id), {'data': scores[0,:,:]})
                    cv2.imwrite('{}/{}.png'.format(edge_dir, img_id), edge_[0,:,:,0] * 255)

                    print_time(timer_step, "\t end step \n\t\t", flg_show=flg_debug)


            self.coord.request_stop()
            self.coord.join(threads)
            print_time(timer_batch, "\nend iteration\n\t", flg_show=flg_debug)