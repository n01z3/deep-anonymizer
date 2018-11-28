import os
import cv2
import numpy as np
import tensorflow as tf

from umeyama import umeyama
import mtcnn_detect_face


def create_mtcnn(sess, model_path):
    if not model_path:
        model_path,_ = os.path.split(os.path.realpath(__file__))
    with tf.variable_scope('pnet2'):
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet = mtcnn_detect_face.PNet({'data':data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet2'):
        data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
        rnet = mtcnn_detect_face.RNet({'data':data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet2'):
        data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
        onet = mtcnn_detect_face.ONet({'data':data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)
    return pnet, rnet, onet


def get_src_landmarks(x0, x1, y0, y1, pnts):
    """
    x0, x1, y0, y1: (smoothed) bbox coord.
    pnts: landmarks predicted by MTCNN
    """
    src_landmarks = [(int(pnts[i + 5][0] - x0),
                      int(pnts[i][0] - y0)) for i in range(5)]
    return src_landmarks


def get_tar_landmarks(img):
    """
    img: detected face image
    """
    ratio_landmarks = [
        (0.31339227236234224, 0.3259269274198092),
        (0.31075140146108776, 0.7228453709528997),
        (0.5523683107816256, 0.5187296867370605),
        (0.7752419985257663, 0.37262483743520886),
        (0.7759613623985877, 0.6772957581740159)
    ]

    img_size = img.shape
    tar_landmarks = [(int(xy[0] * img_size[0]),
                      int(xy[1] * img_size[1])) for xy in ratio_landmarks]
    return tar_landmarks


def landmarks_match_mtcnn(src_im, src_landmarks, tar_landmarks):
    """
    umeyama(src, dst, estimate_scale)
    landmarks coord. for umeyama should be (width, height) or (y, x)
    """
    src_size = src_im.shape
    src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
    tar_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
    M = umeyama(np.array(src_tmp), np.array(tar_tmp), True)[0:2]
    result = cv2.warpAffine(src_im, M, (src_size[1], src_size[0]),
                            borderMode=cv2.BORDER_REPLICATE)
    return result


def process_mtcnn_bbox(bboxes, im_shape):
    """
    output bbox coordinate of MTCNN is (y0, x0, y1, x1)
    Here we process the bbox coord. to a square bbox with ordering (x0, y1, x1, y0)
    """
    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bboxes[i, 0:4]
        w, h = int(y1 - y0), int(x1 - x0)
        length = (w + h) / 2
        center = (int((x1 + x0) / 2), int((y1 + y0) / 2))
        new_x0 = np.max([0, (center[0] - length // 2)])  # .astype(np.int32)
        new_x1 = np.min([im_shape[0], (center[0] + length // 2)])  # .astype(np.int32)
        new_y0 = np.max([0, (center[1] - length // 2)])  # .astype(np.int32)
        new_y1 = np.min([im_shape[1], (center[1] + length // 2)])  # .astype(np.int32)
        bboxes[i, 0:4] = new_x0, new_y1, new_x1, new_y0
    return bboxes


def sorting_box_func(a):
    x0, y1, x1, y0, c = a[0], a[1], a[2], a[3], a[4]
    return -((x1 - x0) ** 2 + (y1 - y0) ** 2)
