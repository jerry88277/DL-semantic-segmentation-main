# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 01:58:24 2021

@author: JerryDai
"""

import os, pickle, time, torch, cv2, random, shutil
import smpgit.segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as albu
from utils import *
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
target_classes = ["void", "road", "lanemarks", "curb", "person", "rider",
           "vehicles", "bicycle","motorcycle", "traffic_sign"]
#ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
ACTIVATION = 'sigmoid'
DEVICE = 'cuda:1'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(target_classes), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# PATH_x = './rgb_images/'
# PATH_y = './WoodScape_ICCV19/semantic_annotations/semantic_annotations/gtLabels/'
# fns_x = os.listdir(PATH_x)
# fns_y = os.listdir(PATH_y)
# fns_x.sort()
# fns_y.sort()

best_model = torch.load(r'D:\NCKU\Class In NCKU\DeepLearning\HW7\DL-semantic-segmentation-main\output\25-01-20-10efficientnet-b4_bs=4_epochs=1\best_model.pth')    

# dataset_nonarg_tr = Dataset(PATH_x + 'tr/', PATH_y + 'tr/', classes=CLASSES)

# dataset_tr = Dataset(
#     PATH_x + 'tr/', PATH_y + 'tr/', classes=CLASSES,
#     augmentation=get_training_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn)
# )

# dataset_va = Dataset(
#     PATH_x + 'va/', PATH_y + 'va/', classes=CLASSES,
#     augmentation=get_validation_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn)
# )

# dataset_te = Dataset(
#     PATH_x + 'te/', PATH_y + 'te/', classes=CLASSES,
#     augmentation=get_validation_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn)
# )

PATH = r'D:\NCKU\Class In NCKU\DeepLearning\HW7\DL-semantic-segmentation-main\rgb_images(test_set)'
dataset_test_no = Dataset(
    PATH, None, classes=target_classes, CLASSES = target_classes,
)

PATH = r'D:\NCKU\Class In NCKU\DeepLearning\HW7\DL-semantic-segmentation-main\rgb_images(test_set)'
dataset_test = Dataset(
    PATH, None, classes = target_classes, CLASSES = target_classes,
    augmentation = get_validation_augmentation(), 
    preprocessing = get_preprocessing_no_mask(preprocessing_fn)
)

# loader_tr = DataLoader(dataset_tr, batch_size=8, num_workers=12)
# loader_va = DataLoader(dataset_va, batch_size=1, num_workers=4)
# loader_te = DataLoader(dataset_te, batch_size=1, num_workers=4)
# loader_test = DataLoader(dataset_test, batch_size=1, num_workers=4)

i = random.choice(range(1766))
plt.imshow(dataset_test_no[i])
plt.savefig('original4.png')


x = dataset_test[i]
x = torch.tensor(x.reshape((1, 3, 966, 1280)))
x = x.to('cuda')
print(x.shape)
pr_mask = best_model.predict(x)
pr_mask = (pr_mask.squeeze().cpu().numpy().round())
print(pr_mask.shape)
img = pr_mask.argmax(axis=0)
plt.imshow(img)
plt.savefig('segmented4.png')















