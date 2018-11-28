import tempfile

import keras.backend as K

from . import utils

from ..ItemSwap.scripts_faces.networks.faceswap_gan_model import FaceswapGANModel
from ..ItemSwap.scripts_faces.converter.video_converter import VideoConverter
from ..ItemSwap.scripts_faces.detector.face_detector import MTCNNFaceDetector

_GAN_RESOLUTION = 128  # 64x64, 128x128, 256x256

_GAN_ARCH_CONFIG = {
    'IMAGE_SHAPE': (_GAN_RESOLUTION, _GAN_RESOLUTION, 3),
    'use_self_attn': True,
    'norm': 'instancenorm',  # instancenorm, batchnorm, layernorm, groupnorm, none
    'model_capacity': 'standard'
}

_FACE_SWAP_OPTIONS = {
    # ===== Fixed =====
    "use_smoothed_bbox": True,
    "use_kalman_filter": True,
    "use_auto_downscaling": False,
    "bbox_moving_avg_coef": 0.65,
    "min_face_area": 55 * 55,
    #    "IMAGE_SHAPE": model.IMAGE_SHAPE,

    # ===== Tunable =====
    "kf_noise_coef": 3e-3,
    "use_color_correction": "hist_match",
    "detec_threshold": 0.7,
    "roi_coverage": 0.9,
    "enhance": 0.0,
    "output_type": 1,
    "direction": "AtoB",
}


class FaceSwapper:
    def __init__(self, paths_config):
        self._DIR = paths_config['face_swap_dir']

        face_detector = MTCNNFaceDetector(sess=K.get_session(),
                                          model_path=paths_config['mtcnn_weights_dir'])
        gan_model = FaceswapGANModel(**_GAN_ARCH_CONFIG)
        gan_model._load_weights(paths_config['face_swap_gan_weights_dir'])

        self._converter = VideoConverter()
        self._converter.set_face_detector(face_detector)
        self._converter.set_gan_model(gan_model)

        self._options = _FACE_SWAP_OPTIONS
        self._options['IMAGE_SHAPE'] = gan_model.IMAGE_SHAPE

    def transform(self, image):
        _, in_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)
        utils.save_image(in_path, image)

        _, out_path = tempfile.mkstemp(suffix='.png', dir=self._DIR)
        self._converter.convert_image(input_fn=in_path, output_fn=out_path, options=_FACE_SWAP_OPTIONS)

        return utils.load_image(out_path)
