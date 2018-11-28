from __future__ import print_function
import os
import cv2
import tensorflow as tf
from PIL import Image
import scipy.io as sio
from . import utils

# import contextlib
import timeit
# import tqdm 

# @contextlib.contextmanager
# def elapsed_timer():
#     start = timeit.default_timer()
#     elapser = lambda: timeit.default_timer() - start
#     yield lambda: elapser()
#     end = timeit.default_timer()
#     elapser = lambda: end - start


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

    def _build_model(self, tta, need_edges=True):

        if 1.0 in tta: 
            tta.remove(1.0)
            
        tta.insert(0, 1.0)

        # graph definition
        with self.graph_pgn.as_default() as g :
                
            image_rev = tf.reverse(self.image, tf.stack([1]))
            image_batch = tf.stack([self.image, image_rev])
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

                if scale >= 1.0 and need_edges:
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
            self.pred_scores = raw_output_all #tf.reduce_max(raw_output_all, axis=3)
            # raw_output_all = tf.argmax(raw_output_all, axis=3)
            # self.pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.

            # --------------
            #  Do something with edges
            # --------------
            if need_edges:
                raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0) # ???????????????????
                head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
                tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
                raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
                raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
                self.pred_edge = tf.sigmoid(raw_edge_all)
            # res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)


            edge_out2 = tf.reduce_mean(tf.stack(edge_out2_list), axis=0)                     


    def build_model(self, n_class=20, path_model_trained='./checkpoint/CIHP_pgn', tta=[0.50, 0.75, 1.25, 1.50, 1.75], img_size=None, need_edges=True):
        self.n_class = n_class
        self.path_model_trained = path_model_trained

        with self.graph_pgn.as_default() as g :
            # Load reader.
            with tf.name_scope("create_inputs"):

                img_path = tf.placeholder(tf.string, name = 'img_path')
                self.image = utils.image_reader.load_image(img_path, img_size)


        # define PGN graph
        self._build_model(tta, need_edges)

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

    def predict(self, img_path: str):

        # if self
        # parsing_, scores, edge_ = self.sess.run([self.pred_scores, self.pred_edge], {'create_inputs/img_path:0': img_path})
        assert os.path.exists(img_path), img_path
        scores = self.sess.run([self.pred_scores], {'create_inputs/img_path:0': img_path})[0][0]

        return scores
        # print(scores.shape)
        # # parsing_dir = '.'
        # cv2.imwrite('1.png', parsing_[0,:,:,0])
        # sio.savemat('2.mat', {'data': scores[0,:,:]})
        # cv2.imwrite('3.png', edge_[0,:,:,0] * 255)
