from __future__ import print_function,division
import scipy.io as sio
import numpy as np
import torch
from utils import k_folds
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

import os, glob
from collections import OrderedDict
from functools import partial
import numpy as np
import scipy.io

import matplotlib.pyplot as plt
from numpy import *
try:
    from importlib import reload # Python 3
except ImportError:
    pass

import torch

from IPython.core.display import display, HTML, clear_output
# from DeepLearnUtils import  trainimagesource, trainutils, pytorchutils
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, nibabel
import sys, getopt
import PIL
from PIL import Image
import imageio
import scipy.misc
import numpy as np
import glob
from torch.utils import data
import torch
import random
from data.augmentations import Compose, RandomRotate, PaddingCenterCrop
from skimage import transform

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from collections import OrderedDict
import shutil
import os
import pickle
import json
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from data.augmentations import Compose, RandomSizedCrop, AdjustContrast, AdjustBrightness, RandomVerticallyFlip, RandomHorizontallyFlip, RandomRotate, PaddingCenterCrop
import os, glob
import argparse
from PIL import Image


def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample


def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def loadDataCloned(path):
    '''Returns images in HWBDC order, masks in HWBD order, phase in 1B order, and numClones as int.'''
    data = np.load(path)
    # for i in data:
    #     print(i)
    images_LV = data['images_LV']
    endo_LV = data['endo_LV']
    epi_LV = data['epi_LV']
    # phase = data['lv_phase']
    masks = ((epi_LV - endo_LV) != 0).astype(np.int32)
    numClones = int(data['numClones'])

    return images_LV, masks, numClones


class LVQuanDataset(Dataset):

    def __init__(self, root, indices, k_slices = 5, split = 'train', augmentations = None):

        self.root = root

        self.indices = indices
        self.k_slices = k_slices
        self.augmentations =augmentations
        self.split = split

        # self.TRAIN_IMG_PATH = os.path.join(root, 'cardiac-dig.mat')
        # data = sio.loadmat('G:\python_project\datasets\LVQuan18\cardiac_database\cardiac-dig.mat')
        data = sio.loadmat(self.root)
        print("data:",data.keys())
        images_LV = data['images_LV']
        #
        # images_LV_test = data['images_LV_30']
        # print('images_lv_test.shape:', images_LV_test.shape)
        endo_LV = data['endo_LV']
        epi_LV = data['epi_LV']
        lv_phase = data['lv_phase']
        rwt = data['rwt']
        areas = data['areas']
        dims = data['dims']
        masks_LV = ((epi_LV + endo_LV)).astype(np.int8)  ###标签分成3类
        # ratio = data['ratio_resize_inverse']
        # pix = data['pix_spa']
        # ratio1 = np.zeros([1, 2900])
        ratio1 = data['ratio_resize_inverse'].flatten('F')
        ratio = ratio1.reshape([2900, 1])
        ratio =ratio.transpose([1,0])
        pix = np.zeros([2900,1])
        for i in range(145):
            pix[20 * i:20 * i + 20, ] = data['pix_spa'][i,]
        pix =pix.transpose([1,0])

        # print('ratio.shape', ratio.shape)###(1, 2900)
        # print('pix.shape:',pix.shape)###(1, 2900)
        # print('lv_phase.shape:',lv_phase.shape) ###(1, 2900)
        # print('rwt.shape:', rwt.shape)###(6, 2900)
        # print('areas.shape:', areas.shape)###(2, 2900)
        # print('dims.shape:', dims.shape)###(3, 2900)
        # masks_LV = ((epi_LV - endo_LV) != 0).astype(np.int8)  ###只分割心肌和背景


        subjects = 145
        # 20 is frames number
        n = self.k_slices

        # 1 - MRI Left Ventricle
        # Shape: (Total images, depth, H, W)
        imgs = np.zeros([(subjects * 20),n,80,80], dtype='f8')
        masks = np.zeros([(subjects * 20),n,80,80], dtype='f8')

        # print('images.shape:', imgs.shape)
        for i in range(subjects):
            v = images_LV[:, :, 20*i:20*i+20]
            w = masks_LV [:, :, 20*i:20*i+20]
            v_i = np.zeros(20 + ((n//2)*2), dtype=int)
            w_i = np.zeros(20 + ((n//2)*2), dtype=int)

            v_i[0:20+(n//2)] = np.arange(-(n//2),20, dtype=int)
            v_i[20+(n//2):] = np.arange(0,(n//2), dtype=int)
            w_i[0:20+(n//2)] = np.arange(-(n//2),20, dtype=int)
            w_i[20+(n//2):] = np.arange(0,(n//2), dtype=int)
            for j in range(20):
                for k in range(n):
                    imgs[j+20*i, k, :, :] = v[:, :, v_i[j+k]]
                    masks[j+20*i, k, :, :] = w[:, :, v_i[j+k]]

        self.images_LV = torch.from_numpy(imgs[indices, :, :, :])
        self.masks_LV = torch.from_numpy(masks[indices, :, :, :])
        # print('self.image.shape:',self.images_LV.shape)
        # print('self.mask.shape:', self.masks_LV.shape)
        # 2 - Cardiac phase
        self.lv_phase = torch.from_numpy(lv_phase.flatten('C'))
        self.lv_phase = self.lv_phase[self.indices]
        # print('slef.lv_phase_shape:', self.lv_phase.shape)   ########
        # print('self.LV_phase:', self.lv_phase)
        # 3 - RWT (6 indices)
        self.rwt = torch.from_numpy(rwt)
        self.rwt = self.rwt[:, self.indices]
        # print('slef.rwt_shape:', self.rwt.shape)
        # print('self.rwt:', self.rwt)

        # 4 - Areas (cavity and myocardium)
        self.areas = torch.from_numpy(areas)
        self.areas = self.areas[:, self.indices]
        # print('slef.areas_shape:', self.areas.shape)
        # print('self.areas:', self.areas)

        # 5 - Dims
        self.dims = torch.from_numpy(dims)
        self.dims = self.dims[:, self.indices]
        # print('slef.dims_shape:', self.dims.shape)
        # print('self,dims', self.dims)
        # 6- ratio
        self.ratio = torch.from_numpy(ratio)
        self.ratio = self.ratio[:, self.indices]
        # 7- pix
        self.pix = torch.from_numpy(pix)
        self.pix = self.pix[:, self.indices]

    def __getitem__(self, index):
        # print('self.image_LV.len:', len(self.images_LV))
        # print('index:', index)
        image_lv = self.images_LV[index,:, :, :]
        mask_lv = self.masks_LV[index, :, :, :]
        # print('image_lv.shape:', image_lv.shape)
        # print('mask_lv.shape:', mask_lv.shape)
        # print('image_lv.value:', image_lv)
        # print('mask_lv.value:', mask_lv)

        phase_lv = self.lv_phase[index]
        rwt = self.rwt[:, index]
        areas = self.areas[:, index]
        dims = self.dims[:, index]
        ratio = self.ratio[:, index]
        pix = self.pix[:, index]
        # print('phase.value:', phase_lv)
        # print('rwtvalue:',rwt)
        # print('areas.value:',areas)
        # print('dims.value:', dims)
        # print('ratio.value:',ratio)
        # print('pix.value:',pix)
        img = np.asarray(image_lv)
        seg = np.asarray(mask_lv)
        ####display#######
        for x in range(len(img)):
            label = seg[x]*120
            image = img[x]
            import matplotlib.pyplot as plt
            # print(label.dtype)
            print(image.dtype)
            # plt.imshow(label, cmap='viridis', clim=(0, 255))
            # plt.imshow(image, cmap='viridis')
            plt.imshow(image, cmap='gray')

            plt.axis('off')
            plt.show()
        #####display######
        if self.augmentations is None:
            img_c = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
            seg_c = np.zeros((seg.shape[0], seg.shape[1], seg.shape[2]))
            # print('img_c.shape:', img_c.shape)
            # print('seg_c.shape:', seg_c.shape)
            for z in range(img.shape[0]):
                # if img[z].min() > 0:
                #     img[z] -= img[z].min()
                # img_tmp, seg_tmp = self.augmentations(img[z].astype(np.float64), seg[z].astype(np.uint8))
                img_tmp, seg_tmp = img[z], seg[z]

                img_tmp = augment_gamma(img_tmp)
                # print('img_temp.shape:', img_tmp.shape)
                # print('seg_temp.shape:', seg_tmp.shape)

                mu = img_tmp.mean()
                sigma = img_tmp.std()
                img_tmp = (img_tmp - mu) / (sigma + 1e-10)
                # print('img3.shape:', img_tmp.shape)
                # print('seg3.shape:', seg_tmp.shape)
                img_c[z] = img_tmp
                seg_c[z] = seg_tmp
                # img_c[z] = crop_or_pad_slice_to_size(img_tmp, 256, 256)
                # seg_c[z] = crop_or_pad_slice_to_size(seg_tmp, 256, 256)
                # print('img_c1.shape:', img_c.shape)
                # print('seg_c1.shape:', seg_c.shape)

            img = img_c
            seg = seg_c
            # print('img.shape:', img.shape)
            # print('seg.shape:', seg.shape)

        img = torch.from_numpy(img).float()
        seg = torch.from_numpy(seg).long()
        # print('img_tensor.shape:', img.shape)
        # print('seg_tensor.shape:', seg.shape)
        # print('img_tensor_value:', img)
        # print('seg_tensor_value:', seg)
        # print('data_dict:', data_dict)
        # ####display#######
        # for x in range(len(img)):
        #     label = seg[x]*120
        #     # image = img[x]
        #     import matplotlib.pyplot as plt
        #     print(label.dtype)
        #     # print(image.dtype)
        #     plt.imshow(label, cmap='viridis', clim=(0, 255))
        #     # plt.imshow(image, cmap='viridis')
        #
        #     plt.axis('off')
        #     plt.show()
        # # ######display######
        if self.split == 'train': # 50% chance of deformation
            img_v = np.zeros([img.shape[0], img.shape[1],img.shape[2]], dtype='f8')
            seg_v = np.zeros([seg.shape[0], seg.shape[1],seg.shape[2]], dtype='f8')
            for i in range(img.shape[0]):
                img_a = img[i].double().numpy()
                seg_a = seg[i].double().numpy()
                # print('img_numpy:', img_a.shape)
                # print('seg_numpy:', seg_a.shape)
                if random.uniform(0, 1.0) <= 0.5:
                    if len(img_a.shape) == 2:
                        # print('img_a.shape:', img_a.shape)
                        img_a = np.expand_dims(img_a, axis=2)
                    seg_a = np.expand_dims(seg_a, axis=2)
                    stacked = np.concatenate((img_a, seg_a), axis=2)
                    red = self.random_elastic_deformation(stacked, alpha=500, sigma=20).transpose(2, 0, 1)
                    img_a, seg_a = red[0], red[1]

                    # print('img_train11.shape:', img_a.shape)
                    # print('seg_train11.shape:', seg_a.shape)
                # if img_a.ndim == 2:
                #     img_a = np.expand_dims(img_a, axis=0)
                    # img_a = np.concatenate((img_a, img_a, img_a), axis=0)
                # End Random Elastic Deformation
                img_v[i] = img_a
                seg_v[i] = seg_a
            # print('img_v.shape:', img_v.shape)
            # print('seg_v.shape:', seg_v.shape)
            # ####display#######
            for x in range(len(img_v)):
                # label = seg_v[x]*100
                image = img_v[x]
                import matplotlib.pyplot as plt
                # print(label.dtype)
                print(image.dtype)
                # plt.imshow(label, cmap='viridis', clim=(0, 255))
                plt.imshow(image, cmap='viridis')

                plt.axis('off')
                plt.show()
            # ######display######
            imgss = np.expand_dims(img_v, axis=0)
            segss = seg_v
            # print('imagess.shape:', imgss.shape)
            # print('segss.shape:', segss.shape)
            # print('imagess_value:', imgss)
            # print('segsss_value:', segss)
            d = {"image": torch.from_numpy(imgss).float(),
                 "mask": torch.from_numpy(segss),'phase_lv': phase_lv,
                  'rwt_lv': rwt, 'areas_lv': areas, 'dims_lv': dims, 'pix':pix, 'ratio':ratio}
            # "mask": (torch.from_numpy(seg),
            #          self.mask_to_edges(seg))}"mask": torch.from_numpy(seg)}#########
            # print('d_train:', d)
            return d

        elif self.split == "test":
            # img_w = np.zeros([img.shape[0], img.shape[1], img.shape[2]], dtype='f8')
            #
            # for i in range(img.shape[0]):
            #     img_b = img[i].unsqueeze(0)
            #     img_b = torch.cat([img_b, img_b, img_b], 0)
            #     print('img_val.shape:', img_b.shape)
            #     print('seg_val.shape:', seg.shape)
            #     img_w[i] = img_b
            # img = img_w
            img = img.unsqueeze(0)
            # print('image_test.shape:', img.shape)
            # print('seg_test.shape:', seg.shape)
            d = {"image": img.float(),
                 # "mask": (seg, self.mask_to_edges(seg)),
                 "mask": seg,'phase_lv': phase_lv,'rwt_lv': rwt,
                 'areas_lv': areas, 'dims_lv': dims, 'pix':pix, 'ratio':ratio}
            return d
            # image = img.float()
            # mask = seg
            #
            # return image, mask

    def random_elastic_deformation(self, image, alpha, sigma, mode='nearest',
                                   random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
    ..  [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        height, width, channels = image.shape

        dx = gaussian_filter(2 * random_state.rand(height, width) - 1,
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter(2 * random_state.rand(height, width) - 1,
                             sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        indices = (np.repeat(np.ravel(x + dx), channels),
                   np.repeat(np.ravel(y + dy), channels),
                   np.tile(np.arange(channels), height * width))

        values = map_coordinates(image, indices, order=1, mode=mode)
        # print('value.shape:', values.shape)
        return values.reshape((height, width, channels))

    def __len__(self):
        return self.images_LV.__len__()


if __name__ == '__main__':

    num_folds = 5

    cont_fold = 1
    DATA_DIR = 'G:\python_project\datasets\LVQuan18\cardiac_database\cardiac-dig.mat'
    # DATA_DIR = 'G:\python_project\datasets\LVQuan18\cardiac_database\lvquan_test_images_30sub.mat'
    # DATA_DIR = 'G:/new_project/18Quan/cardiac-dig_cloned_5.npz'
    # augs = None
    for train_idx, test_idx in k_folds(n_splits = num_folds):
        print('###################### FOLD {} ######################'.format(cont_fold))
        # augs = Compose([PaddingCenterCrop(80), RandomHorizontallyFlip(), RandomVerticallyFlip(), RandomRotate(90)])
        augs = None
        dataset_lv_train = LVQuanDataset(DATA_DIR, indices = train_idx,  split = 'train', k_slices = 5,augmentations= augs)
        dataset_lv_test = LVQuanDataset(DATA_DIR, indices = test_idx,  split = 'test',  k_slices = 5,augmentations= augs)
        dloader = torch.utils.data.DataLoader(dataset_lv_train, batch_size=2)
        for index, batch in enumerate(dloader):
            # print('index:', index)
            # print('index:', index)
            # print('batch.type:', type(batch)
            # print('batch,shape', len(batch))
            img = batch['image']
            mask = batch['mask']
            print('batch_img.shape', img.shape)
            print('batch_mask.shape:', mask.shape)


        # print(mask.shape, img.shape)#, mask[0].max(), mask.min(), img.max(), img.min())
        cont_fold +=1
