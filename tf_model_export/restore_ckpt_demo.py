#coding=utf-8
import tensorflow as tf
import os

sess = tf.Session()
with sess.as_default():
    saver = tf.train.import_meta_graph('./model/ckpt/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/ckpt'))

    input_a = sess.graph.get_operation_by_name('a').outputs[0]
    input_b = sess.graph.get_operation_by_name('b').outputs[0]
    output_c = sess.graph.get_operation_by_name('c').outputs[0]

    a = np.array(
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    )
    b = np.array(
        [
            [6, 7, 8],
            [7, 8, 9]
        ]
    )
    feed_dict = {
        input_a: a,
        input_b: b
    }

    c = sess.run(output_c, feed_dict=feed_dict)
    print(c)


