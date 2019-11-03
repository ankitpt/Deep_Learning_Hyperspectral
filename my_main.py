from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io
# Visualization
import seaborn as sns
import visdom

import os
from utilss import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device 
from my_dataset import get_dataset, HyperA, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model

SAMPLE_PERCENTAGE=0.75
SAMPLING_MODE='random'
center_pixel=True
supervision='full'
patch_size=3

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)




img,gt,labels,ign_labels,rgb,palette=get_dataset("IndianPines", target_folder="./", datasets=DATASETS_CONFIG)

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(labels) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}



train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                 np.count_nonzero(gt)))

#If you want to see train and test datasets ground truth
#display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
#display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")


#        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
 #       if CLASS_BALANCING:
  #          weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
   #         hyperparams['weights'] = torch.from_numpy(weights)
        # Split train set in train/val
train_gt, val_gt = sample_gt(train_gt, 0.95, mode='random')
        # Generate the dataset
train_dataset = HyperA(img, train_gt, patch_size,ign_labels,center_pixel,supervision)
train_loader = data.DataLoader(train_dataset,
                                       batch_size=16,
                                       #pin_memory=hyperparams['device'],
                                       shuffle=True)
val_dataset = HyperA(img, val_gt, patch_size,ign_labels,center_pixel,supervision)
val_loader = data.DataLoader(val_dataset,
                                     #pin_memory=hyperparams['device'],
                                     batch_size=16)
