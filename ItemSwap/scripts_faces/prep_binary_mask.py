import os
import face_alignment
import cv2
import numpy as np
from glob import glob
from pathlib import PurePath, Path
from matplotlib import pyplot as plt
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='data_faces')
    args = parser.parse_args()
    data_folder = args.data_folder

    dir_faceA = f"{data_folder}/facesA/aligned_faces/"
    dir_faceB = f"{data_folder}/facesB/aligned_faces/"
    dir_bm_faceA_eyes = f"{data_folder}/binary_masks/faceA_eyes"
    dir_bm_faceB_eyes = f"{data_folder}/binary_masks/faceB_eyes"

    fns_faceA = glob(f"{dir_faceA}/*.*")
    fns_faceB = glob(f"{dir_faceB}/*.*")

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    Path(f"{data_folder}/binary_masks/faceA_eyes").mkdir(parents=True, exist_ok=True)
    Path(f"{data_folder}/binary_masks/faceB_eyes").mkdir(parents=True, exist_ok=True)

    fns_face_not_detected = []

    for idx, fns in enumerate([fns_faceA, fns_faceB]):
        if idx == 0:
            save_path = dir_bm_faceA_eyes
        elif idx == 1:
            save_path = dir_bm_faceB_eyes

            # create binary mask for each training image
        for fn in fns:
            raw_fn = PurePath(fn).parts[-1]

            x = plt.imread(fn)
            x = cv2.resize(x, (256, 256))
            preds = fa.get_landmarks(x)

            if preds is not None:
                preds = preds[0]
                mask = np.zeros_like(x)

                # Draw right eye binary mask
                pnts_right = [(preds[i, 0], preds[i, 1]) for i in range(36, 42)]
                hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
                mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

                # Draw left eye binary mask
                pnts_left = [(preds[i, 0], preds[i, 1]) for i in range(42, 48)]
                hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
                mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

                # Draw mouth binary mask
                # pnts_mouth = [(preds[i,0],preds[i,1]) for i in range(48,60)]
                # hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
                # mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)

                mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
                plt.imsave(fname=f"{save_path}/{raw_fn}", arr=mask, format="jpg")
            else:
                mask = np.zeros_like(x)
                #             print(f"No faces were detected in image '{fn}''")
                fns_face_not_detected.append(fn)

                p = Path(fn)
                p.unlink()
    # remove only from aligned_face folder
    # val on raw face folder

    num_faceA = len(glob(dir_faceA+"/*.*"))
    num_faceB = len(glob(dir_faceB+"/*.*"))

    print("Number of processed images: " + str(num_faceA + num_faceB))
    print("Number of image(s) with no face detected: " + str(len(fns_face_not_detected)))
