import os
import glob
import tempfile
import argparse
from collections import namedtuple

import numpy as np
import cv2
import tensorflow as tf
import neuralgym as ng

from . import utils
from ..style_transfer.segmentation import SegmentationModel
from ..style_transfer.segmentation import config as segmentation_config
from ..style_transfer.fast_style_transfer.evaluate import ffwd_to_img
from ..ItemSwap.convert_background import run as run_background_converter
from ..ItemSwap.scripts_background.options import test_options as test_background_options
from ..hair.inpaint_model import InpaintCAModel


DEVICE = '/gpu:0'


class SegmentationStyleTransfer:
    def __init__(self, paths_config):
        ng.get_gpus(1)
        self.model_inpaint = InpaintCAModel()
        self.checkpoint_dir_inpaint = 'app/hair/release_celeba_hq_256'

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        import pdb; pdb.set_trace()
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(self.checkpoint_dir_inpaint, from_name)
            assign_ops.append(tf.assign(var, var_value))
        self.sess.run(assign_ops)
        print('Model loaded.')
        
        self._DIR = paths_config['style_transfer_dir']
        self._segmentation_model = SegmentationModel(paths_config['segmentation_weights_dir'])

        self._cloths_style_transfer_weights_dir = paths_config['cloths_style_transfer_weights_dir']
        self._background_style_transfer_weight_dir = paths_config['background_style_transfer_weight_dir']
        

    def transform(self, image, params) -> np.ndarray:
        segmentation_mask = self._get_segmentation_mask(image)
        found_classes = self._get_found_cloths_classes(segmentation_mask)

        if not self._check_if_found_anything(found_classes, params.keys()):
            return image

        transformed_images = dict()
        for style in set(params.values()):
            transformed_images[style] = self._get_cloths_style_transfer(image, style)
        import pdb; pdb.set_trace()
        hair_mask = self._get_hair_mask(segmentation_mask)
        image = self._transform_hair(image, hair_mask)
        
        res, _ = self._morph_transforms_on_params(initial_image=image,
                                               transformed_images=transformed_images,
                                               segmentation_mask=segmentation_mask,
                                               params=params)
       
        return res

        # num_styles = 7
        #
        # res = np.zeros(shape=(num_styles * image.shape[0], len(found_classes) * image.shape[1], 3), dtype=np.uint8)
        #
        # for i, style in enumerate(('jeans', 'leopard', 'udnie', 'scream', 'la_muse', 'wave', 'rain_princess')):
        #     for j, seg_cls in enumerate(found_classes):
        #
        #         transferred_image = self._get_cloths_style_transfer(image, style)
        #         if image.shape != transferred_image.shape:
        #             transferred_image = cv2.resize(transferred_image, tuple(reversed(image.shape[:2])))
        #
        #         morphed, _ = self._morph_transforms(initial_image=image.copy(),
        #                                             transformed_image=transferred_image,
        #                                             segmentation_mask=segmentation_mask,
        #                                             areas=[seg_cls])
        #
        #         morphed = cv2.putText(morphed, seg_cls, (0, morphed.shape[0] - 10),
        #                               cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255))
        #
        #         res[i * image.shape[0]: (i + 1) * image.shape[0],
        #         j * image.shape[1]: (j + 1) * image.shape[1]] = morphed
        #
        # res = cv2.resize(res, (int(res.shape[1] / 3.), int(res.shape[0] / 3.)))
        #
        # print('AAAA INIT SHAPE {}   RES SHAPE {}'.format(image.shape, res.shape))
        # return res

        # if cloths:
        #     style_transfer_image = self._get_cloths_style_transfer(image)
        #     modified_image, _ = self._morph_transforms(
        #         initial_image=modified_image,
        #         transformed_image=style_transfer_image,
        #         segmentation_mask=segmentation_mask,
        #         areas=segmentation_config.SegmentationClassNames.CLOTHS
        #     )
        #
        # if background:
        #     style_transfer_image = self._get_background_style_transfer(image)
        #     modified_image, _ = self._morph_transforms(
        #         initial_image=modified_image,
        #         transformed_image=style_transfer_image,
        #         segmentation_mask=segmentation_mask,
        #         areas=segmentation_config.SegmentationClassNames.BACKGROUND
        #     )
        #
        # return modified_image

    def _check_if_found_anything(self, found_classes, required_classes):
        for cls in required_classes:
            if cls in found_classes:
                return True
        return False

    def _get_found_cloths_classes(self, mask):
        found = []
        for index, class_name in enumerate(segmentation_config.SegmentationClassNames.ALL):
            if class_name not in segmentation_config.SegmentationClassNames.CLOTHS:
                continue
            if (mask != index).all():
                continue
            found.append(class_name)
        return found
    
    def _get_hair_mask(self, mask):
        for index, class_name in enumerate(segmentation_config.SegmentationClassNames.ALL):
            if сlass_name == 'Hair':
                return mask == index
        return np.zeros_like(mask)
    
    def _transform_hair(self, image, mask):
        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = self.model_inpaint.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)

        result = sess.run(output)[0]
        return result

    def _get_segmentation_mask(self, image):
        _, in_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)
        utils.save_image(in_path, image)
        mask, _ = self._segmentation_model.predict(in_path)
        return mask

    def _get_cloths_style_transfer(self, image, style):
        _, in_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)
        utils.save_image(in_path, image)

        if style == 'jeans':
            checkpoint_dir = '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/jeans1/'
        elif style == 'leopard':
            checkpoint_dir = '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/leopard'
        elif style == 'crocodile':
            checkpoint_dir = '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/crocodile1'
        else:
            checkpoint_dir = os.path.join(self._cloths_style_transfer_weights_dir, '{}.ckpt'.format(style))

        _, out1_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)
        ffwd_to_img(
            in_path=in_path, out_path=out1_path,
            checkpoint_dir=checkpoint_dir,
            device=DEVICE
        )
        res = utils.load_image(out1_path)
        if image.shape != res.shape:
            res = cv2.resize(res, tuple(reversed(image.shape[:2])))
        return res

    def _get_background_style_transfer(self, image):
        in_dir = tempfile.mkdtemp(dir=self._DIR)
        _, in_image_path = tempfile.mkstemp(suffix='.png', dir=in_dir)
        utils.save_image(in_image_path, image)

        out_dir = tempfile.mkdtemp(dir=self._DIR)

        options = namedtuple('Options', [])
        options.dataroot = in_dir
        options.batch_size = 1
        options.loadSize = 720
        options.name = 'rick_morty'
        options.model = 'cycle_gan'
        options.checkpoints_dir = self._background_style_transfer_weight_dir
        options.results_dir = out_dir
        options.resize_or_crop = 'none'
        options.isTrain = False

        default_parser = test_background_options.TestOptions().initialize(argparse.ArgumentParser())

        for option_name in test_background_options.ALL_OPTIONS:
            if hasattr(options, option_name):
                continue
            setattr(options, option_name, default_parser.get_default(option_name))
            # print('Setting options.{} = {}'.format(option_name, default_parser.get_default(option_name)))

        run_background_converter(options)

        out_paths = glob.glob(os.path.join(out_dir, '*'))
        assert len(out_paths) == 1, out_paths
        return utils.load_image(out_paths[0])

    @staticmethod
    def _morph_transforms(initial_image, transformed_image,
                          segmentation_mask, areas):
        h, w = initial_image.shape[:2]
        transformed_image = cv2.resize(transformed_image, (w, h))
        assert transformed_image.shape == initial_image.shape, \
            '{} vs {}'.format(transformed_image.shape, initial_image.shape)

        result = initial_image.copy()
        affected_mask = np.zeros(shape=result.shape[:2], dtype=bool)
        for index, class_name in enumerate(segmentation_config.SegmentationClassNames.ALL):
            if class_name not in areas:
                continue
            this_class_mask = segmentation_mask == index
            result[this_class_mask] = transformed_image[this_class_mask]
            affected_mask[this_class_mask] = True
        return result, affected_mask

    @staticmethod
    def _morph_transforms_on_params(initial_image, transformed_images,
                                    segmentation_mask, params):

        result = initial_image.copy()
        affected_mask = np.zeros(shape=result.shape[:2], dtype=bool)

        for index, class_name in enumerate(segmentation_config.SegmentationClassNames.ALL):
            if class_name not in params.keys():
                continue
            this_class_mask = segmentation_mask == index
            this_class_styled_image = transformed_images[params[class_name]]
            assert this_class_styled_image.shape == initial_image.shape, \
                '{} vs {}'.format(this_class_styled_image.shape, initial_image.shape)

            result[this_class_mask] = this_class_styled_image[this_class_mask]

            affected_mask[this_class_mask] = True
        return result, affected_mask
