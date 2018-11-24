import argparse
from keras import backend as K
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt

from images_from_video_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
WEIGHTS_PATH = "weights_faces/mtcnn_weights/"


def process_image(input_img, folder, data_folder, num):
    global pnet, rnet, onet
    assert folder in ['A', 'B']

    minsize = 30  # minimum size of face
    detec_threshold = 0.7
    threshold = [0.6, 0.7, detec_threshold]  # three steps's threshold
    factor = 0.709  # scale factor

    faces, pnts = mtcnn_detect_face.detect_face(
        input_img, minsize, pnet, rnet, onet, threshold, factor)
    faces = process_mtcnn_bbox(faces, input_img.shape)
    faces = sorted(faces, key=sorting_box_func)

    for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces[:1]):
        det_face_im = input_img[int(x0):int(x1), int(y0):int(y1), :]

        # get src/tar landmarks
        src_landmarks = get_src_landmarks(x0, x1, y0, y1, pnts)
        tar_landmarks = get_tar_landmarks(det_face_im)

        # align detected face
        aligned_det_face_im = landmarks_match_mtcnn(
            det_face_im, src_landmarks, tar_landmarks)

        fname = f"{data_folder}/faces{folder}/aligned_faces/img_{num}" \
                f"face{str(idx)}.jpg"
        plt.imsave(fname, aligned_det_face_im, format="jpg")
        fname = f"{data_folder}/faces{folder}/raw_faces/img_{num}" \
                f"face{str(idx)}.jpg"
        plt.imsave(fname, det_face_im, format="jpg")

        bm = np.zeros_like(aligned_det_face_im)
        h, w = bm.shape[:2]
        bm[int(src_landmarks[0][0] - h / 15):int(src_landmarks[0][0] + h / 15),
        int(src_landmarks[0][1] - w / 8):int(src_landmarks[0][1] + w / 8), :] = 255
        bm[int(src_landmarks[1][0] - h / 15):int(src_landmarks[1][0] + h / 15),
        int(src_landmarks[1][1] - w / 8):int(src_landmarks[1][1] + w / 8), :] = 255
        bm = landmarks_match_mtcnn(bm, src_landmarks, tar_landmarks)
        fname = f"{data_folder}/faces{folder}/binary_masks_eyes/img_{num}" \
                f"face{str(idx)}.jpg"
        plt.imsave(fname, bm, format="jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('input_path')
    parser.add_argument('--data_folder', default='data_faces')
    args = parser.parse_args()

    folder = args.folder
    input_path = args.input_path
    data_folder = args.data_folder

    sess = K.get_session()

    with sess.as_default():
        global pnet, rnet, onet
        pnet, rnet, onet = create_mtcnn(sess, WEIGHTS_PATH)

    # global pnet, rnet, onet

    pnet = K.function([pnet.layers['data']], [pnet.layers['conv4-2'], pnet.layers['prob1']])
    rnet = K.function([rnet.layers['data']], [rnet.layers['conv5-2'], rnet.layers['prob1']])
    onet = K.function([onet.layers['data']], [onet.layers['conv6-2'], onet.layers['conv6-3'],
                                              onet.layers['prob1']])

    Path(f"{data_folder}/faces{folder}/aligned_faces").mkdir(parents=True, exist_ok=True)
    Path(f"{data_folder}/faces{folder}/raw_faces").mkdir(parents=True, exist_ok=True)
    Path(f"{data_folder}/faces{folder}/binary_masks_eyes").mkdir(parents=True, exist_ok=True)

    for num, p in enumerate(glob(input_path + '/*')):
        img = cv2.imread(p)[:, :, ::-1]
        process_image(img, folder, data_folder, num)
