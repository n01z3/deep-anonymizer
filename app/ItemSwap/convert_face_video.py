import os
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import argparse
from .scripts_faces.networks.faceswap_gan_model import FaceswapGANModel
from .scripts_faces.converter.video_converter import VideoConverter
from .scripts_faces.detector.face_detector import MTCNNFaceDetector

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

K.set_learning_phase(0)
RESOLUTION = 128  # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, 256"

# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm"  # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard"  # standard, lite
mtcnn_weights_dir = "weights_faces/mtcnn_weights/"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fn')
    parser.add_argument('output_fn')
    parser.add_argument('--start_time', default=15)
    parser.add_argument('--end_time', default=25)
    parser.add_argument('--weights_path',
                        default='weights_faces_sobchak2all/gan_models/backup_iter40000/')

    args = parser.parse_args()
    input_fn = args.input_fn
    output_fn = args.output_fn
    start_time = args.start_time
    end_time = args.end_time
    weights_path = args.weights_path

    model = FaceswapGANModel(**arch_config)
    model._load_weights(path=weights_path)

    fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)
    vc = VideoConverter()
    vc.set_face_detector(fd)
    vc.set_gan_model(model)

    options = {
        # ===== Fixed =====
        "use_smoothed_bbox": True,
        "use_kalman_filter": True,
        "use_auto_downscaling": False,
        "bbox_moving_avg_coef": 0.65,
        "min_face_area": 55 * 55,
        "IMAGE_SHAPE": model.IMAGE_SHAPE,
        # ===== Tunable =====
        "kf_noise_coef": 3e-3,
        "use_color_correction": "hist_match",
        "detec_threshold": 0.7,
        "roi_coverage": 0.9,
        "enhance": 0.0,
        "output_type": 1,
        "direction": "AtoB",
    }

    duration = (int(start_time), int(end_time))
    print('Duration', duration)
    vc.convert(input_fn=input_fn, output_fn=output_fn, options=options,
               duration=duration)  # TODO: make with batch
