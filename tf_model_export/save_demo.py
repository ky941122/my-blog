#coding=utf-8
import tensorflow as tf
import os
import shutil

sess = tf.Session()
with sess.as_default():
    a = tf.placeholder(shape=[None], dtype=tf.float32, name="a")
    b = tf.placeholder(shape=[None], dtype=tf.float32, name="b")
    c = tf.subtract(a, b, name="c")

    model_version = 1
    export_path = os.path.join("./model", str(model_version))
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_input_a = tf.saved_model.utils.build_tensor_info(a)
    tensor_input_b = tf.saved_model.utils.build_tensor_info(b)
    tensor_output_c = tf.saved_model.utils.build_tensor_info(c)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input_a': tensor_input_a,
                    "input_b": tensor_input_b,
                    },
            outputs={'output_c': tensor_output_c,
                     },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'substract_demo': prediction_signature
        }
    )
    builder.save()




