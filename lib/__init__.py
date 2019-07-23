# import tensorflow as tf
# import numpy as np
# pt = tf.placeholder('uint8',shape=[None,1])
# pv = tf.constant([[1,2,5,6],[9,8,5,2]])
# z = [[2],[5],[5],[7]]
# j = pt.shape
# batch = tf.shape(pt)[0]
# print(batch)
# som = tf.reshape(pv,[batch,2])
# with tf.Session() as s:
#     s.run(tf.global_variables_initializer())
# #     s.run(k,{pt:pv})
#     yv = s.run(som,{pt:z})
#     print(yv)
# t1 = [[[1,2], [3,4]],
#       [[5,6], [7,8]]]
# print(np.shape(t1))