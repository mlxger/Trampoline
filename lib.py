import math
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple
from fvcore.nn.squeeze_excitation import SqueezeExcitation


datasetpath = "output_frames_train_abnormal"
outputpath = "Features"
pretrainedpath = "X3D_M_extract_features.pth"
frequency = 16
batch_size = 16
sample_mode = "oversample"

if __name__ == "__main__":
    a = np.load("F:\\ShanghaiTech\\Extract\\Features\\train_abnormal\\01_0014.npy")
    b = np.load("F:\\ShanghaiTech\\Extract\Features\\train_abnormal\\01_0063.npy")

    print(a.shape)
    print(b.shape)