# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/7/17
# __file__ = utils
# __desc__ =
import numpy as np
import tensorflow as tf
import cv2

########################
# np版
########################
def xywh2xyxy(dboxes: np.ndarray):
    tl = dboxes[:,:2] - dboxes[:,2:]//2
    rb = dboxes[:,:2] + dboxes[:,2:]//2
    return np.concatenate([tl,rb],axis=1)


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


def draw_rectangle(src_img, labels, conf, locations, label_map):
    '''
    src_img : 要画框的图片
    labels : 物体得标签,数字形式
    conf : 该处有物体的概率
    locations : 坐标
    label_map : 将labels映射回类别名字

    return
        画上框的图片
    '''
    num_obj = len(labels)
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    img = src_img.copy()
    for i in range(num_obj):
        tl = tuple(locations[i][:2])
        br = tuple(locations[i][2:])
        lbt = label_map[labels[i]]

        cv2.rectangle(img,tl,
                      br,COLORS[i%3],2)
        cv2.putText(img,lbt,tl,FONT, 1, (255, 255, 255), 2)
    img = img[:, :, ::-1]

    return img