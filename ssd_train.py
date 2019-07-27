# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/7/19
# __file__ = ssd_train
# __desc__ =
import os
import numpy as np
import tensorflow as tf
from lib.ssd_anchors import MultiBoxGenerger as MBG
from tensorflow.python import debug as tf_dbg
from lib.ssd_gen_data2 import gen_
from lib.ssd_losses import ssd_loss
from lib.ssd_model import SSD_ as SSD
from config import SSD_config as sc

DATA_FORMAT = "NHWC"
# SSD Network flags
tf.app.flags.DEFINE_float("loss_alpha", 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float("neg_ratio", 3., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float('match_threshold', 0.5, "Matching threshold in the loss function.")
# General flags
tf.app.flags.DEFINE_string("train_dir", '/tmp/ssdmodel',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer("gpu_core", 0, "gpu core which will be used")
tf.app.flags.DEFINE_integer("num_epoch", 2, "epoch number")
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 1,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')
# Optimization Flags.
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0e-19, 'Epsilon term for the optimizer.')
# learning rate flags
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'step_per_decay', 30.0,
    'Number of epochs after which learning rate decays.')
# dataset flags
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_300_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', 300, 'Train image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)

    sc.im_shape = FLAGS.train_image_size
    tf.summary.scalar("img shape",tf.constant(sc.im_shape))
    mb = MBG(sc)
    print('mb.default_boxes',mb.default_boxes)
    n_anchors = len(mb.default_boxes)
    data_gen = gen_(FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.train_image_size)
    imdir = os.path.join(FLAGS.dataset_dir, "JPEGImages")
    total_file = int(len(os.listdir(imdir)) / FLAGS.batch_size)

    device = f"/cpu:{FLAGS.gpu_core}" if FLAGS.gpu_core else f"/cpu:0"
    global_steps = tf.Variable(0, trainable=False)
    start_lr = FLAGS.learning_rate
    lr = tf.train.exponential_decay(start_lr, global_steps,
                                    FLAGS.step_per_decay, FLAGS.learning_rate_decay_factor)

    bglocaliztion = tf.placeholder('float32', name='bboxes', shape=[FLAGS.batch_size, n_anchors, 4])
    bgconfo = tf.placeholder('uint8', name='labels', shape=[FLAGS.batch_size, n_anchors, 1])

    bgconf = tf.one_hot(bgconfo, depth=FLAGS.num_classes)
    b, _, _, c = bgconf.shape

    bgconf = tf.reshape(bgconf, [b, -1, c])

    ssd = SSD(FLAGS.num_classes, FLAGS.batch_size, FLAGS.train_image_size, device)
    loc_preds, cls_preds = ssd.build()  # op
    for k,v in ssd.end_point.items():
        tf.summary.histogram(k,v)

    cls_loss, loc_loss = ssd_loss(loc_preds, cls_preds, bglocaliztion, bgconf, FLAGS.loss_alpha, FLAGS.neg_ratio)
    # FIXME:为什么这种方式不行
    # loc_loss = tf.losses.get_losses("locs")
    # cls_loss = tf.losses.get_losses("conf")
    total_loss = tf.add_n([cls_loss, loc_loss], name="totalloss")
    tf.summary.scalar('cls_loss',cls_loss)
    tf.summary.scalar('loc_loss',loc_loss)
    tf.summary.scalar('total_loss',total_loss)


    optimzer = tf.train.RMSPropOptimizer(learning_rate=lr,
                                         decay=FLAGS.rmsprop_decay,
                                         momentum=FLAGS.rmsprop_momentum,
                                         epsilon=FLAGS.opt_epsilon)
    gradient, variable = zip(*optimzer.compute_gradients(total_loss))
    gradient, _ = tf.clip_by_global_norm(gradient, 1.25)
    # for o in tf.trainable_variables():
    #     print(o)
    train_op = optimzer.apply_gradients(zip(gradient, variable), global_steps)

    count_step = 0
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(log_device_placement=False,
                            gpu_options=gpu_option)
    saver = tf.train.Saver(max_to_keep=5,
                           write_version=2,
                           pad_step_number=False)
    model_dir = os.path.join(FLAGS.train_dir + "model")
    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
    model_dir = os.path.join(model_dir,FLAGS.model_name)
    log_dir = os.path.join(FLAGS.train_dir +"log")
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    # TODO:supervisor
    with tf.Session(config=config) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        summary_merge = tf.summary.merge_all()  # 合并所有summary
        f_summary = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess,coord)

        print('start training.....')
        while count_step < total_file:
            gb_image, gblocaliztion, gbconf = data_gen.__next__()
            # print(gblocaliztion)
            # print(gbconf)
            tgbloc, tgbconf = [], []
            for gbl, gbc in zip(gblocaliztion, gbconf):
                tgl, tgc = mb.encode(gbl, gbc,0.3)
                tgbloc.append(tgl)
                tgbconf.append(tgc)
            gblocaliztion = np.stack(tgbloc)
            gbconf = np.stack(tgbconf)
            if not np.any(gbconf):
                continue
            if not np.any(gblocaliztion):
                continue
            print('img',np.any(gb_image))
            if gbconf.ndim != 3 or gblocaliztion.ndim != 3:
                print(gbconf)
                print(gblocaliztion)
                raise ValueError('       ')
            tloss, lloss, closs, _ = sess.run(
                [total_loss, loc_loss, cls_loss, train_op],
                feed_dict={
                    ssd.inputs: gb_image,
                    bgconfo: gbconf,
                    bglocaliztion: gblocaliztion
                }
            )
            if np.isnan(tloss):
                print('exit -----')
                print(gbconf,np.any(gbconf))
                print(gblocaliztion,np.any(gblocaliztion))
                exit(-1)
            # summary_tmp = sess.run(summary_merge,feed_dict={
            #     ssd.inputs: gb_image,
            #     bgconfo: gbconf,
            #     bglocaliztion: gblocaliztion
            # })
            # f_summary.add_summary(summary_tmp,global_step=count_step)

            if count_step % FLAGS.log_every_n_steps == 0:
                print(f'step:{count_step + 1}  total_loss:{tloss}  '
                      f'loc_loss:{lloss}  cls_loss:{closs}')
            if count_step % FLAGS.save_interval_secs == 0:
                saver.save(sess, model_dir, global_steps, )
            count_step += 1

        # except tf.errors.OutOfRangeError:
        #     print("read done!")
        # finally:
        #     coord.request_stop()

        print("训练结束")


if __name__ == '__main__':
    tf.app.run()
