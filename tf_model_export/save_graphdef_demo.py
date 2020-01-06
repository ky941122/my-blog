#coding=utf-8
import tensorflow as tf

sess = tf.Session()
with sess.as_default():

    a = tf.placeholder(shape=[None], dtype=tf.float32, name="a")
    b = tf.placeholder(shape=[None], dtype=tf.float32, name="b")
    c = tf.subtract(a, b, name="c")

    tf.train.write_graph(sess.graph_def, "./model/graphDef", "model.pbtxt", as_text=True)

    # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["c"])
    # with tf.gfile.GFile("./model/graphDef/model.pb", "wb") as model_f:
    #     model_f.write(output_graph_def.SerializeToString())