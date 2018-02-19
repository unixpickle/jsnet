import tensorflow as tf

v1 = tf.Variable([0.5, -0.3, 0.2, 0.7, -0.8, 0.9])
v2 = tf.Variable([0.1, -0.7, -0.1, 0.3, 0.5, -0.6])
upstream = tf.constant([1, 2, 3, 4, 5, 6], dtype=v1.dtype)
result = v1 * (v2 / (v1 - v2)) + v1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gradients(tf.reduce_sum(result * upstream), [v1, v2])))
