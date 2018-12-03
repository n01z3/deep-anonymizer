# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import round2nearest_multiple
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata
import cv2
from scipy.misc import imread, imresize
from torchvision import transforms
assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'
parser = argparse.ArgumentParser()
# Path related arguments
parser.add_argument('--test_folder', required=True)
# parser.add_argument('--model_path', required=True, help='folder to model path')
parser.add_argument('--model_path', help='folder to model path', default='baseline-resnet50_dilated8-ppm_bilinear_deepsup')
parser.add_argument('--suffix', default='_epoch_20.pth', help="which snapshot to load")
# Model related arguments
parser.add_argument('--arch_encoder', default='resnet50_dilated8', help="architecture of net_encoder")
parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup', help="architecture of net_decoder")
parser.add_argument('--fc_dim', default=2048, type=int, help='number of features between encoder and decoder')
# Data related arguments
parser.add_argument('--num_val', default=-1, type=int, help='number of images to evalutate')
parser.add_argument('--num_class', default=150, type=int, help='number of classes')
parser.add_argument('--batch_size', default=1, type=int, help='batchsize. current only supports 1')
parser.add_argument('--imgSize', default=[300, 400, 500, 600], nargs='+', type=int, help='list of input image sizes.' 'for multiscale testing, e.g. 300 400 500')
parser.add_argument('--imgMaxSize', default=1000, type=int, help='maximum input image size of long edge')
parser.add_argument('--padding_constant', default=8, type=int, help='maxmimum downsampling rate of the network')
parser.add_argument('--segm_downsampling_rate', default=8, type=int, help='downsampling rate of the segmentation label')
# Misc arguments
parser.add_argument('--result', default='.', help='folder to output visualization results')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id for evaluation')
args = parser.parse_args()

# absolute paths of model weights
args.weights_encoder = os.path.join(args.model_path, 'encoder' + args.suffix)
args.weights_decoder = os.path.join(args.model_path, 'decoder' + args.suffix)
args.arch_encoder = 'resnet50_dilated8'
args.arch_decoder = 'ppm_bilinear_deepsup'
args.fc_dim = 2048
assert os.path.exists(args.weights_encoder) and os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

if not os.path.isdir(args.result):
    os.makedirs(args.result)

torch.cuda.set_device(args.gpu_id)


def overlay(img, pred_color, blend_factor=0.3):
    edges = cv2.Canny(pred_color,20,40)
    edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=1)
    out = (1-blend_factor)*img + blend_factor * pred_color
    edge_pixels = (edges==255)
    new_color = [0,0,255]
    for i in range(0,3):
        timg = out[:,:,i]
        timg[edge_pixels]=new_color[i]
        out[:,:,i] = timg
    return out

# Network Builders
builder = ModelBuilder()
net_encoder = builder.build_encoder(arch=args.arch_encoder, fc_dim=args.fc_dim, weights=args.weights_encoder)
net_decoder = builder.build_decoder(arch=args.arch_decoder, fc_dim=args.fc_dim, num_class=args.num_class, weights=args.weights_decoder, use_softmax=True)
crit = nn.NLLLoss(ignore_index=-1)
input_fns = [os.path.join(args.test_folder,f) for f in os.listdir(args.test_folder)]
output_fns = [os.path.join(args.result,f[0:-3]+"pgm") for f in os.listdir(args.test_folder)]
output_vis_fns = [os.path.join(args.result,"vis_" + f) for f in os.listdir(args.test_folder)]
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()
colors = loadmat('data/color150.mat')['colors']

transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])
feed_dict = {}
for f,of,ovf in zip(input_fns,output_fns,output_vis_fns):
    print("Input: " + f)
    print("Output: " + of)
    img = imread(f, mode='RGB')
    img = img[:, :, ::-1]  # BGR to RGB!!!
    ori_height, ori_width, _ = img.shape

    img_resized_list = []
    for this_short_size in args.imgSize:
        scale = this_short_size / float(min(ori_height, ori_width))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        target_height = round2nearest_multiple(target_height, args.padding_constant)
        target_width = round2nearest_multiple(target_width, args.padding_constant)
        img_resized = cv2.resize(img.copy(), (target_width, target_height))
        img_resized = img_resized.astype(np.float32)
        img_resized = img_resized.transpose((2, 0, 1))
        img_resized = transform(torch.from_numpy(img_resized))
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    input = dict()
    input['img_ori'] = img.copy()
    input['img_data'] = [x.contiguous() for x in img_resized_list]

    segSize = (img.shape[0],img.shape[1])

    with torch.no_grad():
        pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        for timg in img_resized_list:
            feed_dict = dict()
            feed_dict['img_data'] = timg.cuda()
            feed_dict = async_copy_to(feed_dict, args.gpu_id)
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            pred = pred + pred_tmp.cpu() / len(args.imgSize)

        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0))

    # prediction
    pred_color = colorEncode(preds, colors)

    # aggregate images and save
    blended = overlay(img,pred_color)
    im_vis = np.concatenate((img, pred_color, blended), axis=1).astype(np.uint8)
    cv2.imwrite(ovf, im_vis)
    cv2.imwrite(of, preds)

