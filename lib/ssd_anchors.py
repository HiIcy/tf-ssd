# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/7/17
# __file__ = layers
# __desc__ =
import itertools
import math
import tensorflow as tf
from config import SSD_config
import numpy as np
from .utils import tf_box_iou, tf_xywh2xyxy, tf_ssd_bboxes_encode,tf_xyxy2xywh


def ssd_size_bounds_to_values(n_fea_layer, size_bound: tuple, img_shape=(300, 300)):
    """
    # 这种计算方式参考SSD的Caffe源码
    """
    assert img_shape[0] == img_shape[1]
    img_size = img_shape[0]
    max_ratio = int(size_bound[0] * 100)
    min_ratio = int(size_bound[1] * 100)
    # FAQ: n_feature_layer=6? 第一层单独设置
    step = int(math.floor((max_ratio - min_ratio) / (n_fea_layer - 2)))
    sizes = [[img_size * size_bound[1] / 2, img_size * size_bound[1]]]
    for ratio in range(min_ratio, max_ratio, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    print(sizes)
    return sizes


class MultiBoxGenerger(object):
    def __init__(self, sizes=None, sc=SSD_config):
        self.variance = sc.variance
        default_boxes = []  # [[x,y,w,h]
        self.imshape = sc.im_shape
        self.sizes = sc.anchor_sizes if sizes == None else sizes
        for i in range(len(sc.grids)):
            # TODO:用meshgrid
            for v, u in itertools.product(range(sc.grids[i][0]), repeat=2):
                # x,y= np.meshgrid(range(sc.grids[i]),range(sc.grids[i]))
                cx = (u + 0.5) * sc.steps[i] / self.imshape
                cy = (v + 0.5) * sc.steps[i] / self.imshape
                cs = sc.anchor_sizes[i]
                default_boxes.append((cx, cy, cs / self.imshape, cs / self.imshape))
                cs = math.sqrt(sc.anchor_sizes[i] * sc.anchor_sizes[i + 1])
                default_boxes.append((cx, cy, cs / self.imshape, cs / self.imshape))
                for v in sc.ratios[i]:
                    default_boxes.append((cx, cy, sc.anchor_sizes[i] * math.sqrt(v) / self.imshape,
                                          sc.anchor_sizes[i] / math.sqrt(v) / self.imshape))
        default_boxes = np.clip(default_boxes, 0, 1)  # 归一化
        self.default_boxes = tf.convert_to_tensor(default_boxes,
                                                  dtype="float32")

    def encode(self, bboxes, labels, threshold=0.5):
        # TODO:用纯tensorflow 实现
        """
        type:
        #REW:给每个anchor分配一个bbox 加label
        """
        if len(bboxes) == 0:
            return (tf.zeros(self.default_boxes.shape, dtype=np.float32),
                    tf.zeros(self.default_boxes.shape[:1], dtype=np.float32))
        ious = tf_box_iou(tf_xywh2xyxy(self.default_boxes), bboxes)  # (m,n)
        pidx = tf.argmax(ious, axis=1)  # 得到bboxes的索引(m)
        iou = tf.reduce_max(ious, axis=1)

        bboxes[pidx, 0] = (bboxes[pidx, 0] + bboxes[pidx, 2]) * 0.5  # 取出m个n范围里的值
        bboxes[pidx, 1] = (bboxes[pidx, 1] + bboxes[pidx, 3]) * 0.5
        bboxes[pidx, 2] = bboxes[pidx, 2] - bboxes[pidx, 0]
        bboxes[pidx, 3] = bboxes[pidx, 3] - bboxes[pidx, 1]
        # 边界框的预测值 的转换值，我们称上面这个过程为边界框的编码
        loc_gt = np.zeros((len(self.default_boxes), 4), dtype=np.float32)
        loc_gt[:, 0] = (bboxes[pidx, 0] - self.default_boxes[:, 0]) / (self.default_boxes[:, 2] * self.variance[0])
        loc_gt[:, 1] = (bboxes[pidx, 1] - self.default_boxes[:, 1]) / (self.default_boxes[:, 3] * self.variance[1])
        loc_gt[:, 2] = np.log(bboxes[pidx, 2] / self.default_boxes[:, 2]) / self.variance[2]
        loc_gt[:, 3] = np.log(bboxes[pidx, 3] / self.default_boxes[:, 3]) / self.variance[3]
        labeler = labels[pidx]
        # 设置为0，通过这一步我们拥有了这个标签后就能知道哪些anchor是正样本
        conf = 1 + labeler
        conf[iou < threshold] = 0
        return loc_gt, conf.astype(np.int32)

    def tf_encode(self, bboxes, labels, threshold=0.5):
        # 后面有用batch函数，所以这里单个就行
        if bboxes.get_shape().as_list()[0] == 0:
            return (tf.zeros(self.default_boxes.shape, dtype=tf.float32),
                    tf.zeros(self.default_boxes.shape[:1], dtype=tf.uint8))
        # print(bboxes.get_shape().as_list())
        b,_,_ = bboxes.get_shape().as_list()
        loc_record = []
        conf_record = []
        for i in range(b):
            ious = tf_box_iou(tf_xywh2xyxy(self.default_boxes), bboxes[i])  # (m,n)
            pidx = tf.argmax(ious, axis=1)  # 得到bboxes的索引(m)
            iou = tf.reduce_max(ious, axis=1)

            bboxs = tf_xyxy2xywh(tf.gather(bboxes[i],pidx))
            # stack 会增加维度，所以不必用:切片
            loc_gt = tf.stack([
                    tf.div(bboxs[:,0] - self.default_boxes[:,0],(self.default_boxes[:,2]*self.variance[0])),
                    tf.div(bboxs[:,1] - self.default_boxes[:,1],(self.default_boxes[:,3]*self.variance[1])),
                    tf.log(bboxs[:,2]/self.default_boxes[:,2]) / self.variance[2],
                    tf.log(bboxs[:,3]/self.default_boxes[:,3]) / self.variance[3]
            ],axis=-1)
            labeler = tf.gather(labels[i],pidx)
            conf = 1 + labeler
            tzero = tf.zeros_like(conf)
            conditon = iou > threshold  # (m,)
            conf = tf.where(conditon,tzero,conf)
            loc_record.append(loc_gt)
            conf_record.append(conf)
        bloc_gt = tf.stack(loc_record)
        bconf = tf.stack(conf_record)
        return bloc_gt,bconf


    def decode(self, p_loc):
        loc_gt = np.zeros((len(self.default_boxes), 4), dtype=np.float32)
        loc_gt[:, 0] = self.variance[0] * p_loc[:, 0] * self.default_boxes[:, 2] + self.default_boxes[:, 0]
        loc_gt[:, 1] = self.variance[1] * p_loc[:, 1] * self.default_boxes[:, 3] + self.default_boxes[:, 1]
        loc_gt[:, 2] = self.default_boxes[:, 2] * np.exp(self.variance[2] * p_loc[:, 2])
        loc_gt[:, 3] = self.default_boxes[:, 3] * np.exp(self.variance[3] * p_loc[:, 3])
        # REW:同时操作x，y坐标
        loc_gt[:, :2] -= loc_gt[:, 2:] / 2
        loc_gt[:, 2:] += loc_gt[:, :2]  # REW:变成坐标形式

        return loc_gt

    def tf_decode(self,p_loc):  # 针对单张图片 TODO:batch支持
        loc_pr = tf.stack([
            self.variance[0] * p_loc[:, 0] * self.default_boxes[:, 2] + self.default_boxes[:, 0],
            self.variance[1] * p_loc[:, 1] * self.default_boxes[:, 3] + self.default_boxes[:, 1],
            self.default_boxes[:, 2] * np.exp(self.variance[2] * p_loc[:, 2]),
            self.default_boxes[:, 3] * np.exp(self.variance[3] * p_loc[:, 3])
        ])


class TfMultiBoxGenerger(object):
    def __init__(self, sizes=None, sc=SSD_config):
        self.variance = sc.variance
        self.imshape = sc.im_shape
        self.n_class = sc.n_class
        self.sizes = sc.anchor_sizes if sizes is None else sizes
        self.anchors = self._ssd_anchors_all_layers([sc.im_shape, sc.im_shape],
                                                    sc.grids,
                                                    self.sizes,
                                                    sc.ratios,
                                                    sc.steps,
                                                    )

    def _ssd_anchor_one_layer(self, img_shape,
                              feat_shape,
                              sizes,
                              ratios,
                              step,
                              offset=0.5,
                              dtype=np.float32):
        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        y = (y.astype(dtype) + offset) * step / img_shape[0]
        x = (x.astype(dtype) + offset) * step / img_shape[1]

        # Expand dims to support easy broadcasting.
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)

        # Compute relative height and width.
        # Tries to follow the original implementation of SSD for the order.
        num_anchors = len(sizes) + len(ratios)
        h = np.zeros((num_anchors,), dtype=dtype)
        w = np.zeros((num_anchors,), dtype=dtype)
        # Add first anchor boxes with ratio=1.
        h[0] = sizes[0] / img_shape[0]
        w[0] = sizes[0] / img_shape[1]
        di = 1
        if len(sizes) > 1:
            h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
            w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
            di += 1
        for i, r in enumerate(ratios):
            h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
            w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
        return y, x, h, w

    def _ssd_anchors_all_layers(self, img_shape,
                                layers_shape,
                                anchor_sizes,
                                anchor_ratios,
                                anchor_steps,
                                offset=0.5,
                                dtype=np.float32):
        """Compute anchor boxes for all feature layers.
        """
        layers_anchors = []
        for i, s in enumerate(layers_shape):
            anchor_bboxes = self._ssd_anchor_one_layer(img_shape, s,
                                                       anchor_sizes[i],
                                                       anchor_ratios[i],
                                                       anchor_steps[i],
                                                       offset=offset, dtype=dtype)
            layers_anchors.append(anchor_bboxes)
        return layers_anchors

    def tf_encode(self, bboxes, labels):
        return tf_ssd_bboxes_encode(labels, bboxes,
                                    self.anchors,
                                    self.n_class,)
