#coding=utf-8
import tensorflow as tf
import os
import shutil

sess = tf.Session()
with sess.as_default():
    a = tf.placeholder(shape=[None, 3], dtype=tf.float32, name="a")
    b = tf.placeholder(shape=[None, 3], dtype=tf.float32, name="b")
    temp1 = tf.subtract(a, b)
    temp2 = tf.keras.layers.Dense(units=16)(temp1)
    c = tf.identity(temp2, name="c")

    sess.run(tf.global_variables_initializer())

    model_version = 2
    export_path = os.path.join("./model", str(model_version))
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
    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    #
    # tensor_input_a = tf.saved_model.utils.build_tensor_info(a)
    # tensor_input_b = tf.saved_model.utils.build_tensor_info(b)
    # tensor_output_c = tf.saved_model.utils.build_tensor_info(c)
    #
    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'input_a': tensor_input_a,
    #                 "input_b": tensor_input_b,
    #                 },
    #         outputs={'output_c': tensor_output_c,
    #                  },
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    #
    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         'substract_demo': prediction_signature
    #     }
    # )
    # builder.save()

    # tf.train.write_graph(sess.graph_def, export_path, "model.pb", as_text=False)
    #
    # graph_def = tf.get_default_graph().as_graph_def()
    # # 保存指定的节点，并将节点值保存为常数
    # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ["c"])
    # # 将计算图写入到模型文件中
    # model_f = tf.gfile.GFile("./4/model.pb", "wb")
    # model_f.write(output_graph_def.SerializeToString())
