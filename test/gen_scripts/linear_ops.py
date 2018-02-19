import tensorflow as tf

v1 = tf.Variable([[0.5, 0.2, -0.2], [-0.8, -0.3, 1]])
v2 = tf.Variable([[0.3, -0.2, 0.4, -0.4], [0.1, -0.7, 0.3, -0.2], [0.4, 0.6, -0.2, -0.1]])
upstream = tf.constant([[1, -0.7, -0.5, 0.5], [0.3, -0.3, 0.4, -1]])
result = tf.matmul(v1, v2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(result))
    print(sess.run(tf.gradients(tf.reduce_sum(result * upstream), [v1, v2])))
