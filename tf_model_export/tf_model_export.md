#Tensorflow模型固化与跨语言调用

相信使用过tensorflow的人都接触过它的模型保存功能，这在日常中的使用预训练模型、测试已有模型以及继续训练当前模型都是必备的技巧。但工作学习中的大多数时候我们使用到的保存模型的api都是：

```angular2
saver = tf.train.Saver()
saver.save()
```

运行之后就可以得到四个文件：checkpoint、model.ckpt.data、model.ckpt.index、model.ckpt.meta，分别保存了checkpoint文件信息、参数值信息、参数名称信息以及图结构信息。

使用这种方式进行模型的保存读取对于实验过程来说完全足够了，但若是我们实验得到了一个不错的模型，现在想要将它部署上线，checkpoint就显得不够灵活了。这里主要存在着这样几个缺点：首先最明显的，checkpoint文件只能用于python环境下的tensorflow中；其次，对于tensorflow版本也有一定的要求，版本号跨度太大的框架之间无法互相读取模型（比如旧版本中一些op被弃用的情况）。

在说本篇分享的正题之前，最后再插入一句题外话。tensorflow一直以来以其僵化、难用出名，但这么久以来仍能牢牢占据大量市场份额而没有被pytorch干掉，就是依仗其整套完整的工业化生态圈。因此虽然静态图（tf-1.x而论）如此恶心人，但tensorflow仍能和pytorch互相瓜分工业圈与学术圈。

而这篇分享主要依赖的背景，便是tensorflow为了解决线上与本地框架版本不同这一痛点而推荐使用的tf serving作为线上模型部署方式。（但考虑到部署tf serving的教程需要很长的篇幅，而大部分小伙伴的实验环境中并没有这项服务，为了方便大家运行例子，因此后续的代码例子中将不会使用tf serving来进行。大家只需要知道，后面要介绍的模型导出方法依照官方的意图是推荐与tf serving配合使用的。）

接下来便是本文的正题所在：

### **使用tf.SavedModelBuilder固化模型**

首先说好处：

1、使用这种方法得到的模型文件可以跨语言调用（下面给出基于java的例子）

2、使用这种方法得到的模型文件中将变量固化为了常量，减少了模型大小，且对于框架版本的依赖也消失了。

3、方便模型版本控制。（指的是模型迭代的版本，而不是框架的版本）

接着上例子：

```angular2
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
    export_path = os.path.join(".", str(model_version))
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
```

为了例子的简单易懂，本文放弃了官网的mnist例子，使用一个简单的减法作为例子。正所谓一法通万法通，看懂了这个例子后，所有基于tf-1.x且使用placeholder作为模型读入的例子都是一个道理。(使用estimator的模型的保存方法其实就是给原本模型读入数据的地方开个placeholder的口子即可，考虑到使用estimator的场合不多见，这里就不展开讲了。)

代码中的a、b、c三个op即为我们平时模型中的输入和输出，而这些数量都是可变的，你可以有多个输入，也可以有多个输出。
