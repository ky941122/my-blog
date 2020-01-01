#coding=utf-8
import tensorflow as tf
import os
import numpy as np


# with tf.Session(graph=tf.Graph()) as sess:
#     saver = tf.train.import_meta_graph('./ckpt/model.meta')
#     # saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))
#     saver.restore(sess, "./ckpt/model")
#
#     input_a = sess.graph.get_operation_by_name('a').outputs[0]
#     input_b = sess.graph.get_operation_by_name('b').outputs[0]
#
#     output_c = sess.graph.get_operation_by_name('c').outputs[0]
#
#     a = np.array(
#         [
#             [1, 2, 3],
#             [4, 5, 6]
#         ]
#     )
#     b = np.array(
#         [
#             [6, 7, 8],
#             [7, 8, 9]
#         ]
#     )
#     feed_dict = {
#         input_a: a,
#         input_b: b
#     }
#
#     c = sess.run(output_c, feed_dict=feed_dict)
#     print(c.shape)


# print("#"*50)


model_filename = "./4/model.pb"
with tf.gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def)
with tf.Session() as sess:
    print([n.name for n in tf.get_default_graph().as_graph_def().node])

    input_a = sess.graph.get_tensor_by_name('import/a:0')
    input_b = sess.graph.get_tensor_by_name('import/b:0')

    output_c = sess.graph.get_tensor_by_name('import/c:0')

    a = np.array(
        [
            [1,2,3],
            [4,5,6]
        ]
    )
    b = np.array(
        [
            [6,7,8],
            [7,8,9]
        ]
    )
    feed_dict = {
        input_a: a,
        input_b: b
    }


    c = sess.run(output_c, feed_dict=feed_dict)
    print(c.shape)
    print("trainable:", tf.trainable_variables())
