#coding=utf-8
import tensorflow as tf
import os
import shutil

sess = tf.Session()
with sess.as_default():
    a = tf.placeholder(shape=[None], dtype=tf.float32, name="a")
    b = tf.placeholder(shape=[None], dtype=tf.float32, name="b")
    c = tf.subtract(a, b, name="c")

    model_version = 3
    export_path = os.path.join(".", str(model_version))
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    tf.saved_model.simple_save(sess,
                               export_path,
                               inputs={'input_a': a,
                                       "input_b": b,
                                       },
                               outputs={'output_c': c,
                                        }
                               )





    #
    #
    #
    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    #
    # signature = tf.saved_model.signature_def_utils.predict_signature_def(
    #     inputs={'input_a': a,
    #             "input_b": b,
    #             },
    #     outputs={'output_c': c,
    #              }
    # )
    #
    # builder.add_meta_graph_and_variables(
    #     sess,
    #     tags=[tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         'substract_demo': signature
    #     }
    # )
    # builder.save()
    #


