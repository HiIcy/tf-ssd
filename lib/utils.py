# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/7/17
# __file__ = utils
# __desc__ =
import numpy as np
import tensorflow as tf


########################
# np版
########################
def xywh2xyxy(dboxes: np.ndarray):
    dboxes[:, 0] -= dboxes[:, 2] / 2
    dboxes[:, 1] -= dboxes[:, 3] / 2
    dboxes[:, 2] += dboxes[:, 2] / 2
    dboxes[:, 3] += dboxes[:, 3] / 2
    return dboxes


def tf_xyxy2xywh(dboxes:tf.Tensor):
    xmin = dboxes[:, :1]
    ymin = dboxes[:, 1:2]
    xmax = dboxes[:, 2:3]
    ymax = dboxes[:, 3:]
    cx = (xmin + xmax)*0.5
    cy = (ymin + ymax)*0.5
    w = (xmax - xmin)
    h = (ymax - ymin)
    return tf.concat([cx,cy,w,h],axis=1)


def tf_xywh2xyxy(dboxes:tf.Tensor):
    cx = dboxes[:,:1]
    cy = dboxes[:,1:2]
    w = dboxes[:,2:3]
    h = dboxes[:,3:]
    cx_w = cx - w/2
    cxw = cx+w/2
    cy_h = cy - h/2
    cyh = cy +h/2
    return tf.concat([cx_w,cy_h,cxw,cyh],axis=1)

def box_iou(dboxes, bboxes):
    m = dboxes.shape[0]
    n = bboxes.shape[0]
    # REW：broadcasting,这么做相当于(m,1,1)和(1,n,1)的进行操作,最后得到(m,n,1)大小
    xmin = np.maximum(dboxes[:, None, 0], bboxes[None, :, 0])
    ymin = np.maximum(dboxes[:, None, 1], bboxes[None, :, 1])
    xmax = np.minimum(dboxes[:, None, 2], bboxes[None, :, 2])
    ymax = np.minimum(dboxes[:, None, 3], bboxes[None, :, 3])

    w = np.maximum(xmax - xmin, 0)
    h = np.maximum(ymax - ymin, 0)

    inner = w * h
    d = (dboxes[:, 3] - dboxes[:, 1]) * (dboxes[:, 2] - dboxes[:, 0])  # m
    b = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])  # n
    d = d[:, None]
    b = b[None, :]
    outter = d + b
    # 返回（m，n）
    return inner / (outter - inner)

def tf_box_iou(dboxes,bboxes):
    m = dboxes.shape[0]
    n = bboxes.shape[0]
    # REW：broadcasting,这么做相当于(m,1)和(1,n)的进行操作,tf最后得到(m,n,)大小
    xmin = tf.maximum(dboxes[:, None, 0], bboxes[None, :, 0])
    ymin = tf.maximum(dboxes[:, None, 1], bboxes[None, :, 1])
    xmax = tf.minimum(dboxes[:, None, 2], bboxes[None, :, 2])
    ymax = tf.minimum(dboxes[:, None, 3], bboxes[None, :, 3])
    w = tf.maximum(xmax-xmin,0)
    h = tf.minimum(ymax-ymin,0)
    inner = w*h # (m,n) 每个都是相交叉面积
    d = (dboxes[:, 3] - dboxes[:, 1]) * (dboxes[:, 2] - dboxes[:, 0])  # (m)
    b = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])  # (n)
    d = d[:,None]
    b = b[None,:]
    outter = tf.add(d,b) # (m,n)
    return tf.div(inner,(outter+inner))

def nms(pboxes, scores, threshold=0.4):
    idx = np.argsort(scores)
    pick = []
    while len(idx) > 0:
        ci = idx[-1]  # 当前最大索引
        pick.append(ci)
        if len(idx) == 1:
            break
        idx = idx[:-1]
        box = pboxes[ci].reshape(1, 4)
        ious = box_iou(box, pboxes[idx])  # (1,n)
        idx = np.delete(idx, np.where(ious > threshold)[0])
    return pick


##############################
def detect(loc_pred, cls_pred, nms_threshold, gt_threshold,scope="nms_output"):
    # loc_pred:[-1,4],[-1,20
    cls_pred = cls_pred[:, 1:]  # 排除背景类别
    # 存放最后保留物体的信息,他的坐标,置信度以及属于哪一类
    keep_boxes = []
    keep_labels = []
    classes = tf.argmax(cls_pred,axis=1)+1  # 得到每个框的类别
    scores = tf.reduce_max(cls_pred,axis=1)

    mask = scores > gt_threshold
    keep_confs:tf.Tensor = tf.boolean_mask(scores,mask)
    keep_boxes = tf.boolean_mask(loc_pred,mask)
    keep_labels = tf.boolean_mask(classes,mask)

    # FIXME:nms现在先用tensorflow的
    pick = tf.image.non_max_suppression(keep_boxes,tf.reshape(keep_confs,[-1]),200,nms_threshold,name=scope)
    keep_confs = tf.gather(keep_confs,pick)  # 按tensorflow的tensor索引来切片
    keep_labels = tf.gather(keep_labels,pick)
    keep_boxes = tf.gather(keep_boxes,pick)

    return keep_boxes,keep_confs,keep_labels  # [bbox,4,1,1]


#########################
# tf版
#########################
def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size) # [h,w,4]
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        # interscts = intersection_with_anchors(bbox)
        # mask = tf.logical_and(interscts > ignore_threshold,
        #                       label == no_annotation_label)
        # # Replace scores by -1.
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores

def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):

    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        return target_labels, target_localizations, target_scores


def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes

def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes
