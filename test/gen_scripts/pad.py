import tensorflow as tf

image = tf.round(tf.random_normal([2, 4, 2, 3]) * 100) / 100
padded = tf.pad(image, [[0, 0], [1, 2], [3, 4], [0, 0]])
upstream = tf.round(tf.random_normal([2, 7, 9, 3]) * 100) / 100
grad = tf.gradients(tf.reduce_sum(padded * upstream), image)[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run((image, padded, upstream, grad))
    for label, item in zip(['image', 'padded', 'upstream', 'grad'], result):
        print('const %s = new jsnet.Tensor([%s], [%s]);' %
              (label, ', '.join(str(x) for x in item.shape),
               ', '.join('%.3f' % x for x in item.flatten())))
