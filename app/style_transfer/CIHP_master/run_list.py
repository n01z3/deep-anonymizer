import os
# import torch
import numpy as np
# import yaml
from collections import namedtuple
# import torchvision.transforms as transforms
import sys 
# os.
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(FILE_PATH)


import numpy as np
# from PIL import Image
# from image_matting import Matting as MattingInternal
# from pspnet import SqueezePSPNet
# from torch.autograd import Variable
import cv2 
# import matplotlib.pyplot as plt
# from utils import preds2dets
from PGN_single import PGN
from PIL import Image
from utils_ import preds2dets
import os
from utils_ import preds2coloredseg
# import tqdm

# def to_namedtuple(dictionary):
#     return namedtuple('GenericDict', dictionary.keys())(**dictionary)

# def get_image_pil(path):
#     img = Image.open(path).convert('RGB')
#     return img



class_names_ = [ 'Background', 
                'Hat', 
                'Hair', 
                'Glove', 
                'Sunglasses', 
                'Upper clothes',
                'Dress', 
                'Coat', 
                'Socks', 
                'Pants', 
                'Torso skin', 
                'Scarf', 
                'Skirt',
                'Face', 
                'Left arm', 
                'Right arm', 
                'Left leg', 
                'Right leg', 
                'Left shoe', 
                'Right shoe']

class_names = [ 'Background', 
                'Hat', 
                'Hair', 
                'Glove', 
                'Sunglasses', 
                'Upper clothes',
                'Dress', 
                'Coat', 
                'Socks', 
                'Pants', 
                'Torso skin', 
                'Scarf', 
                'Skirt',
                'Face', 
                'Left arm', 
                'Right arm', 
                'Left leg', 
                'Right leg', 
                'Shoes']


def extract_largest_component (image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape, dtype=np.uint8)
    img2[output == max_label] = 255

    return img2


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis)[..., None])
    return e_x / e_x.sum(axis=axis)[..., None]


def get_image_pil(path):
    img = Image.open(path).convert('RGB')
    return img


class DSModel:
   
    def __init__(self, **kwargs):

        model_pgn = PGN()

        model_pgn.build_model(n_class=20, path_model_trained='./checkpoint/CIHP_pgn', tta = [0.75, 0.5], img_size = (512, 512), need_edges=False)

        self.model_pgn = model_pgn
        self.to_return='det'
        self.threshold = 0.7

    def predict(self, img_path, **kwargs):
        img_pil  = get_image_pil(img_path)

        scores = self.model_pgn.predict(img_path) # W x H x 20
            
        probs = np.ascontiguousarray(softmax(scores, axis = 2).transpose(2, 0, 1))

        # Merge Shoes
        probs[-2] += probs[-1]
        props = probs[:-1]
            
        classes_ = np.argmax(probs, 0).astype(np.uint8)

        confidences = [] 
        for i in range(len(class_names)):


            nc = np.sum(classes_ == i) 

            confidence = 0 if nc == 0 else probs[i][classes_ == i].sum() / nc
                
            confidences.append(confidence)


        return preds2coloredseg(probs, img_pil, out_format='gray'), confidences



    def get_return_type(self):
        if self.to_return == 'det':
            return 'dict'
        elif self.to_return == 'seg':
            return 'img_path'
        else:
            assert False





if __name__ == '__main__':
    
    import argparse 
    import os 

    import tqdm
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_to_list', type=str, help='')

    args = parser.parse_args()

    d = DSModel()

    files = np.loadtxt(args.path_to_list, dtype=str)

    for img_path in tqdm.tqdm(files):
        

        if not os.path.exists(img_path):
            print(img_path)
            continue


        OUT_DIR = f'{os.path.dirname(img_path)}/segs'
        if os.path.exists(f'{OUT_DIR}/{os.path.basename(img_path)}_cihp.png'):
            continue


        img_seg_pil, confidences = d.predict(img_path)

        ###
        os.makedirs(OUT_DIR, exist_ok=True)
        ###

        save_path = f'{OUT_DIR}/{os.path.basename(img_path)}_cihp.txt'
        np.savetxt(save_path, confidences)

        save_path = f'{OUT_DIR}/{os.path.basename(img_path)}_cihp.png'
        img_seg_pil.save(save_path)

