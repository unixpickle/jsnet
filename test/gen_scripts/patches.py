import tensorflow as tf

image = tf.round(tf.random_normal([2, 5, 8, 3]) * 100) / 100
patches = tf.extract_image_patches(image, [1, 3, 2, 1], [1, 2, 3, 1], [1, 1, 1, 1], 'VALID')
upstream = tf.round(tf.random_normal(patches.get_shape()) * 100) / 100
grad = tf.gradients(tf.reduce_sum(patches * upstream), image)[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run((image, patches, upstream, grad))
    for label, item in zip(['image', 'patches', 'upstream', 'grad'], result):
        print('const %s = new jsnet.Tensor([%s], [%s]);' %
              (label, ', '.join(str(x) for x in item.shape),
               ', '.join('%.3f' % x for x in item.flatten())))
