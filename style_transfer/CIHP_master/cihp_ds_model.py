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
        
        np.save('/seg_out/cihp_segs.npy', probs)

        if self.to_return == 'seg':
            save_path = 'tmp.png'
            preds2coloredseg(save_path, probs, self.out_format)
            return save_path


        '''
            Detection
        '''
        if self.to_return == 'det':
            return preds2dets(probs, img_pil, class_names, self.threshold)




    def get_return_type(self):
        if self.to_return == 'det':
            return 'dict'
        elif self.to_return == 'seg':
            return 'img_path'
        else:
            assert False
