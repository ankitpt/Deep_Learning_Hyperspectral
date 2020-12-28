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
from my_models import get_model, train, test, save_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



dataset_name="Salinas"

img,gt,labels,ign_labels,rgb,palette=get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG)
N_CLASSES = len(labels)
N_BANDS=img.shape[2]
N_CLASSES=len(labels)
IGNORED_LABELS=ign_labels
CUDA_DEVICE=get_device(-1)



hyperparams_old={'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
             'device': CUDA_DEVICE,'epoch':200,'test_stride':1}

#Select best hyperparams here
#9, 0.0017075474589876183, 256
#[5, 0.0017596788469351823, 192]
hyperparams_old.update({'learning_rate': 0.0017075474589876183, 'patch_size': 9,'filters':256})

CHECKPOINT="C:/Users/17657/Desktop/Boil_03/IE_590/Project/Hyperspectral-Classification/IndianaPines.pth"

model, _, _, hyperparams = get_model("lee", **hyperparams_old)
model.to(CUDA_DEVICE)
model.load_state_dict(torch.load(CHECKPOINT),map_location=torch.device('cpu'))
probabilities = test(model, img, hyperparams)
prediction = np.argmax(probabilities, axis=-1)
