# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/7/12
# __file__ = config
# __desc__ =
import xml.etree.ElementTree as ET
from pathlib import Path
import os


class SSD_config:
    n_anchors = [4, 6, 6, 6, 4, 4]
    feature_layer = ['conv4_3', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11']
    im_shape = 300
    grids = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_sizes = [21, 45, 99, 153, 207, 261, 315]
    ratios = [[2, 5],
              [2, .5, 3, 1. / 3],
              [2, .5, 3, 1. / 3],
              [2, .5, 3, 1. / 3],
              [2, .5],
              [2, .5]]
    steps = [8, 16, 30, 60, 100, 300]
    variance = [0.1, 0.1, 0.2, 0.2]
    n_class = 21
