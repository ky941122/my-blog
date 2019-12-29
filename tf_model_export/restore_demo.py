#coding=utf-8
import tensorflow as tf
import os
import numpy as np

with tf.Session(graph=tf.Graph()) as sess:
    model_version = 1
    export_path = os.path.join(".", str(model_version))

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
    sess.run(tf.global_variables_initializer())

    input_a = sess.graph.get_operation_by_name('a').outputs[0]
    input_b = sess.graph.get_operation_by_name('b').outputs[0]

    output_c = sess.graph.get_operation_by_name('c').outputs[0]

    a = np.array(
        [1,2,3,4,5]
    )
    b = np.array(
        [6,7,8,9,10]
    )
    feed_dict = {
        input_a: a,
        input_b: b
    }

    c = sess.run(output_c, feed_dict=feed_dict)
    print(c)