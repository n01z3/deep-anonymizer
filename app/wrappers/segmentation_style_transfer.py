import argparse
import glob
import os
import tempfile
import time
from collections import namedtuple

import cv2
import numpy as np

from . import utils
from ..ItemSwap.convert_background import run as run_background_converter
from ..ItemSwap.scripts_background.options import test_options as test_background_options
from ..style_transfer.FastPhotoStyle.demo import MyNvidiaWrapper
from ..style_transfer.fast_style_transfer.evaluate import MyWrapper as FSTWrapper
from ..style_transfer.segmentation import SegmentationModel
from ..style_transfer.segmentation import config as segmentation_config
from ..style_transfer.sky_seg.test import SkySegWrapper

DEVICE = '/gpu:0'

NVIDIA_STYLES = [
    'summer',
    'winter',
    'inferno',
    'undead'
]


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


class SegmentationStyleTransfer:
    def __init__(self, paths_config):
        self._DIR = paths_config['style_transfer_dir']
        self._segmentation_model = SegmentationModel(paths_config['segmentation_weights_dir'])

        self._cloths_style_transfer_weights_dir = paths_config['cloths_style_transfer_weights_dir']
        self._background_style_transfer_weight_dir = paths_config['background_style_transfer_weight_dir']

        self._style_transformers = dict()
        self._style_transformers['jeans'] = FSTWrapper(
            '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/jeans1')
        self._style_transformers['leopard'] = FSTWrapper(
            '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/leopard')
        self._style_transformers['crocodile'] = FSTWrapper(
            '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/crocodile1')

        for style in ('udnie', 'scream', 'la_muse', 'wave', 'rain_princess', 'wreck'):
            self._style_transformers[style] = FSTWrapper(
                os.path.join(self._cloths_style_transfer_weights_dir, '{}.ckpt'.format(style)))

        self._cached_segmentations = dict()
        self._cached_styles = dict()
        self._cached_sky_seg = dict()

        self._sky_seg = SkySegWrapper()
        self._nvidia_transfer = MyNvidiaWrapper()

    def transform(self, init_path, image, params) -> np.ndarray:
        if init_path in self._cached_segmentations.keys():
            segmentation_mask, segmentation_probas = self._cached_segmentations[init_path]
            segmetation_mask = segmentation_mask.copy()
            segmentation_probas = segmentation_probas.copy()
        else:
            with Timer('Cloths Segmentation: %f'):
                segmentation_mask, segmentation_probas = self._get_segmentation_mask(image)
            self._cached_segmentations[init_path] = segmentation_mask.copy(), segmentation_probas.copy()

        sky_seg_scores = None
        if 'Background' in params:
            if init_path in self._cached_sky_seg.keys():
                sky_seg_scores = self._cached_sky_seg[init_path].copy()
            else:
                with Timer('Sky Segmentation: %f'):
                    sky_seg_scores = self._sky_seg.run(init_path)
                    self._cached_sky_seg[init_path] = sky_seg_scores.copy()
            print('YYYYYYYYYYYYYYYYYYYYYYYYYY', sky_seg_scores.shape)
        found_classes = self._get_found_cloths_classes(segmentation_mask)

        if not self._check_if_found_anything(found_classes, params.keys()):
            return image

        transformed_images = dict()

        for style in set(params.values()):
            if init_path in self._cached_styles.keys() and style in self._cached_styles[init_path].keys():
                transformed_images[style] = self._cached_styles[init_path][style].copy()
            else:
                with Timer('Style "{}": %f'.format(style)):
                    transformed_images[style] = self._get_cloths_style_transfer(image, style)
                if init_path not in self._cached_styles.keys():
                    self._cached_styles[init_path] = dict()
                self._cached_styles[init_path][style] = transformed_images[style].copy()

        res, _ = self._morph_transforms_on_params(
            initial_image=image,
            transformed_images=transformed_images,
            segmentation_probas=segmentation_probas,
            params=params,
            sky_seg_scores=sky_seg_scores)
        return res

    def _check_if_found_anything(self, found_classes, required_classes):
        for cls in required_classes:
            if cls in found_classes:
                return True
        return False

    def _get_found_cloths_classes(self, mask):
        found = []
        for index, class_name in enumerate(segmentation_config.SegmentationClassNames.ALL):
            if (mask != index).all():
                continue
            found.append(class_name)
        return found

    def _get_segmentation_mask(self, image):
        _, in_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)
        utils.save_image(in_path, image)
        mask, probas = self._segmentation_model.predict(in_path)
        return mask, probas

    def _get_cloths_style_transfer(self, image, style):

        # print('Running for style ', style)

        if style not in NVIDIA_STYLES:
            _, in_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)

            small_image = cv2.resize(image.copy(), (1024, 1024))

            # small_image = image.copy()

            utils.save_image(in_path, small_image)

            # if style == 'jeans':
            #     checkpoint_dir = '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/jeans1/'
            # elif style == 'leopard':
            #     checkpoint_dir = '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/leopard'
            # elif style == 'crocodile':
            #     checkpoint_dir = '/home/ikibardin/Kaggle/fast-style-transfer/my_checkpoints/crocodile1'
            # else:
            #     checkpoint_dir = os.path.join(self._cloths_style_transfer_weights_dir, '{}.ckpt'.format(style))
            #
            _, out1_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)
            # ffwd_to_img(
            #     in_path=in_path, out_path=out1_path,
            #     checkpoint_dir=checkpoint_dir,
            #     device=DEVICE
            # )
            self._style_transformers[style].transform_single(in_path=in_path, out_path=out1_path)

        else:
            _, in_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)
            h, w = image.shape[:2]
            factor = 512. / max(h, w)
            new_h = int(h * factor)
            new_w = int(w * factor)

            small_image = cv2.resize(image, (new_w, new_h))

            utils.save_image(in_path, small_image)

            style_image_path = '/home/ikibardin/deep-anonymizer/app/style_transfer/FastPhotoStyle/images/{}.jpg'.format(
                style)
            _, out1_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)

            self._nvidia_transfer.transform(in_path, style_image_path, out1_path)

        res = utils.load_image(out1_path)
        if image.shape != res.shape:
            res = cv2.resize(res, tuple(reversed(image.shape[:2])))
        assert image.shape == res.shape, '{} vs {}'.format(image.shape, res.shape)
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
                                    segmentation_probas, params, sky_seg_scores):

        result = initial_image.copy()

        for index, class_name in enumerate(segmentation_config.SegmentationClassNames.ALL):
            if class_name not in params.keys():
                continue
            alpha = np.array([segmentation_probas[index]] * 3)
            alpha = np.transpose(alpha, (1, 2, 0))

            if class_name == 'Background':
                assert sky_seg_scores is not None
                sky_seg_scores = np.array([sky_seg_scores] * 3)
                sky_seg_scores = np.transpose(sky_seg_scores, (1, 2, 0))
                if sky_seg_scores.shape != alpha.shape:
                    sky_seg_scores = cv2.resize(sky_seg_scores, tuple(reversed(alpha.shape[:2])),
                                                interpolation=cv2.INTER_AREA)
                alpha = alpha * sky_seg_scores

            if alpha.shape != initial_image.shape:
                alpha = cv2.resize(alpha, tuple(reversed(initial_image.shape[:2])))

            if class_name != 'Background':
                alpha[alpha < 0.5] = 0.
            else:
                alpha[alpha < 0.2] = 0.
            alpha[alpha > 0.7] = 1.

            this_class_styled_image = transformed_images[params[class_name]]

            assert this_class_styled_image.shape == initial_image.shape, \
                '{} vs {}'.format(this_class_styled_image.shape, initial_image.shape)

            result = (result * (1. - alpha) + this_class_styled_image * alpha).astype(np.uint8)

        return result, None
