# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/7/18
# __file__ = ssd_losses
# __desc__ =
import tensorflow as tf


def hard_negtives(labels, logits: tf.Tensor, pos, neg_radio):
    # FIXME:用tensorflow里的实现
    """
    label:(batch,n,21)
    logit:(batch,n,21)
    pos:(batch,n)
    """
    batch, num_anchors, num_class = logits.get_shape().as_list()
    logits = tf.reshape(logits, [-1, num_class])
    labels = tf.reshape(labels, [-1,num_class])

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)  # 每个框都有一个损失
    losses: tf.Tensor = tf.reshape(losses, [batch, num_anchors])  # [batch,n]
    losses = losses * (1-pos)
    # REW:第一次排序 得到正常排序完的位置
    loss_idx = tf.argsort(losses, axis=1, direction="DESCENDING")  # [batch,n]
    # REW：然后排序得到降排序但是按0索引来的位置 并且索引放在对应元素下面
    rank = tf.argsort(loss_idx, axis=1)  # (batch, n)
    n_pos = tf.reduce_sum(pos, axis=1, keepdims=True)
    # 不超过最大
    n_neg = tf.clip_by_value(neg_radio * n_pos, -1, pos.shape.as_list()[1] - 1)  # 保证负样本小于总样本数
    # _,neg_mask = tf.nn.top_k(losses,n_neg,sorted=False) # (batch,n_neg)
    neg_mask = tf.less(tf.cast(rank,'float32'), n_neg)  # (batch,n)
    return neg_mask


def my_smooth_l1_loss(bbox_pred,bbox_target,n_batch,alpha=1):
    box_diff = bbox_pred - bbox_target
    abs_box_diff = tf.abs(box_diff)
    smoothl1_mask = tf.to_float(tf.less(abs_box_diff,1))
    in_loss_box = tf.pow(box_diff,2)*0.5*smoothl1_mask + (abs_box_diff - 0.5)*(1-smoothl1_mask)
    outloss_box = alpha*in_loss_box
    loss = tf.reduce_mean(tf.reduce_sum(outloss_box,axis=[1]),name="locs") # FIXME:tf.div?
    return loss


def ssd_loss(loc_pre: tf.Tensor, cls_pre, loc_gt, cls_gt, alpha=1,neg_radio=3.):
    """
    loc_pre:(batch, anchor_num, 4)
    cls_pre:(batch, anchor_num, num_classes)
    neg_radio:
    n_class:
    loc_gt: (batch, anchor_num, 4)
    cls_gt:(batch,anchor,num_classes) onehot
    :return:
    """
    # print(loc_pre.shape,cls_pre.shape)
    # print(loc_gt.shape,cls_gt.shape)
    n_batch = loc_pre.get_shape().as_list()[0]
    # 因为xij存在，所以位置误差仅针对正样本进行计算
    glabel = tf.argmax(cls_gt,axis=-1)  # (batch,anchor)
    pos_mask = glabel > 0  # 挑选出正样本 (batch,anchor)
    pos_idx = tf.cast(pos_mask,tf.float32)
    loc_pre_pos = tf.boolean_mask(loc_pre,pos_mask) # REW:果然可以广播;广播可以忽略最后一个维度
    loc_gt_pos = tf.boolean_mask(loc_gt,pos_mask)
    with tf.name_scope("localization"):
        loc_loss = my_smooth_l1_loss(loc_pre_pos,loc_gt_pos,n_batch,alpha)
        tf.losses.add_loss(loc_loss)
    logits = tf.stop_gradient(cls_pre)  # 只是作负样本选择，所以不计算梯度
    labels = tf.stop_gradient(cls_gt)
    neg_mask = hard_negtives(labels,logits,pos_idx,neg_radio)
    # FIXME:分开来得到pos，neg
    conf_p = tf.boolean_mask(cls_pre,tf.logical_or(pos_mask,neg_mask))
    target = tf.boolean_mask(cls_gt,tf.logical_or(pos_mask,neg_mask))
    # 下面几行都是对的
    # cls_pre_pos = tf.boolean_mask(cls_pre,pos_mask)  # 应该有广播
    # ×cls_pre_neg = cls_pre*neg_idx  # 不能广播
    # cls_gt_pos = tf.boolean_mask(cls_gt,pos_mask)
    # FIXME:分开来用softmaxce;
    with tf.name_scope("cross_entropy"):
        # 交叉熵都会减少一个维度
        cls_loss = tf.nn.softmax_cross_entropy_with_logits_v2(target,conf_p)
        N = tf.reduce_sum(pos_idx)  # 最后除以正样本
        tf.assert_greater(N,tf.cast(0.,tf.float32))
        cls_loss = tf.div(tf.reduce_sum(cls_loss),N,name="conf")
        tf.losses.add_loss(cls_loss)
    return cls_loss,loc_loss



