#coding=utf-8
import tensorflow as tf
import numpy as np
from google.protobuf import text_format

model_filename = "./model/graphDef/model.pbtxt"
with tf.gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    file_content = f.read()
    text_format.Merge(file_content, graph_def)

tf.import_graph_def(graph_def, name='import')

with tf.Session() as sess:
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    print("variable:", tf.global_variables())

    input_a = sess.graph.get_tensor_by_name('import/a:0')
    input_b = sess.graph.get_tensor_by_name('import/b:0')

    output_c = sess.graph.get_tensor_by_name('import/c:0')

    a = np.array(
        [1,2,3,4]
    )
    b = np.array(
       [5,6,7,8]
    )
    feed_dict = {
        input_a: a,
        input_b: b
    }

    c = sess.run(output_c, feed_dict=feed_dict)
    print(c.shape)
    print("variable:", tf.global_variables())
