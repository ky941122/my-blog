#coding=utf-8
import tensorflow as tf
import os

sess = tf.Session()
with sess.as_default():
    a = tf.placeholder(shape=[None, 3], dtype=tf.float32, name="a")
    b = tf.placeholder(shape=[None, 3], dtype=tf.float32, name="b")
    temp1 = tf.subtract(a, b)
    temp2 = tf.keras.layers.Dense(units=16)(temp1)
    c = tf.identity(temp2, name="c")

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join("./model", "ckpt/model"))


