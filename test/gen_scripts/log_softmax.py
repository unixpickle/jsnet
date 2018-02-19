import tensorflow as tf

values = tf.abs(tf.round(tf.random_normal([4, 2, 3]) * 100) / 100) + 0.01
outputs = tf.nn.log_softmax(values)
upstream = tf.round(tf.random_normal([4, 2, 3]) * 100) / 100
grad = tf.gradients(tf.reduce_sum(outputs * upstream), values)[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run((values, outputs, upstream, grad))
    for label, item in zip(['values', 'outputs', 'upstream', 'grad'], result):
        print('const %s = new jsnet.Tensor([%s], [%s]);' %
              (label, ', '.join(str(x) for x in item.shape),
               ', '.join('%.5f' % x for x in item.flatten())))
