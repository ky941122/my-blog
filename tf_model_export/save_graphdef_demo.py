#coding=utf-8
import tensorflow as tf
import os
import shutil

sess = tf.Session()
with sess.as_default():

    a = tf.placeholder(shape=[None], dtype=tf.float32, name="a")
    b = tf.placeholder(shape=[None], dtype=tf.float32, name="b")
    c = tf.subtract(a, b, name="c")

    # graph_def = tf.get_default_graph().as_graph_def()
    # 保存指定的节点，并将节点值保存为常数
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["c"])
    # 将计算图写入到模型文件中
    with tf.gfile.GFile("./model/4/model.pb", "wb") as model_f:
        model_f.write(output_graph_def.SerializeToString())