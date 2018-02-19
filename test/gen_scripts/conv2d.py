import tensorflow as tf

image = tf.round(tf.random_normal([2, 5, 8, 4]) * 100) / 100
filters = tf.round(tf.random_normal([2, 3, 4, 5]) * 100) / 100
outputs = tf.nn.conv2d(image, filters, [1, 2, 1, 1], 'VALID')
upstream = tf.round(tf.random_normal(outputs.get_shape()) * 100) / 100
img_grad, filter_grad = tf.gradients(tf.reduce_sum(outputs * upstream), [image, filters])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run((image, filters, outputs, upstream, img_grad, filter_grad))
    for label, item in zip(['image', 'filters', 'outputs', 'upstream', 'imageGrad', 'filterGrad'], result):
        print('const %s = new jsnet.Tensor([%s], [%s]);' %
              (label, ', '.join(str(x) for x in item.shape),
               ', '.join('%.3f' % x for x in item.flatten())))
