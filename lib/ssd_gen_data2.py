# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/7/22
# __file__ = ssh_gen_data2
# __desc__ =
import os
import itertools
import tensorflow as tf
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from .ssd_preprocessing import preproc_for_train

class_labels = ['boat', 'sheep', 'sofa', 'tvmonitor', 'diningtable', 'car', 'pottedplant', 'horse', 'dog', 'aeroplane',
                'cow', 'person', 'bottle', 'motorbike', 'bicycle', 'train', 'chair', 'cat', 'bird', 'bus']
_R_MEAN = 123
_G_MEAN = 117
_B_MEAN = 104


def _generate_path_list(data_dir):
    imgList = []
    annoList = []
    imgdir = os.path.join(data_dir, "JPEGImages")
    annodir = os.path.join(data_dir, "Annotations")
    for img, anno in zip(os.listdir(imgdir), os.listdir(annodir)):
        pimg = os.path.join(imgdir, img)
        panno = os.path.join(annodir, anno)
        imgList.append(pimg)
        annoList.append(panno)
    return imgList, annoList


def gen_(datadir, batch_size, resize_shape, class_labels=class_labels, test_size_ratio=0.2):
    imgList, annoList = _generate_path_list(datadir)
    iaList = zip(imgList, annoList)
    zipd = itertools.cycle(iaList)

    def _convert_annotation(anno_file):
        try:
            tree = ET.parse(anno_file)
            root = tree.getroot()
            boxes = []
            labels = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in class_labels:
                    continue
                cls_id = class_labels.index(cls)
                bndbox = obj.find('bndbox')
                b = [
                    float(bndbox.find(t).text) - 1
                    for t in ['xmin', 'ymin', 'xmax', 'ymax']
                ]
                boxes.append(b)
                labels.append([cls_id])
        except:
            raise ValueError("读取annotation文件出错")
        else:
            return np.array(boxes), np.array(labels)

    def _normalize_img(img):
        image = cv2.imread(img, cv2.COLOR_BGR2RGB)
        image: np.ndarray = cv2.resize(image, (resize_shape, resize_shape))
        image = image.astype(np.float32)
        image[:, :, 0] -= _R_MEAN
        image[:, :, 1] -= _G_MEAN
        image[:, :, 2] -= _B_MEAN

        return image

    while True:
        train_images = []
        train_boxes = []
        train_labels = []
        while len(train_images) < batch_size:
            data = zipd.__next__()
            img, anno = data
            # img = _normalize_img(img)
            boxes, labels = _convert_annotation(anno)
            if len(boxes) == 0:
                print(1)
                continue
            img = cv2.imread(img, cv2.COLOR_BGR2RGB)
            img, boxes, labels = preproc_for_train(img, boxes, labels, 300, (104, 117, 123))
            # 过滤0值图片
            if not np.any(img):
                continue

            train_images.append(img)
            train_boxes.append(boxes)
            train_labels.append(labels)
        yield train_images, train_boxes, train_labels
