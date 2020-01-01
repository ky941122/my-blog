# Tensorflow模型导出与跨语言调用

相信使用过tensorflow的人都接触过它的模型保存功能，这在日常中的使用预训练模型、测试已有模型以及继续训练现有模型都是必备的技巧。但工作学习中的大多数时候我们使用到的保存模型的api都是：

```angular2
saver = tf.train.Saver()
saver.save()
```

运行之后就可以得到四个文件：checkpoint、model.ckpt.data、model.ckpt.index、model.ckpt.meta，分别保存了checkpoint文件信息、参数值信息、参数名称信息以及图结构信息。

使用这种方式进行模型的保存读取对于实验过程来说完全足够了，但若是我们实验得到了一个不错的模型，现在想要将它部署上线，checkpoint就显得不够灵活了。这里主要存在着这样几个缺点：首先最明显的，checkpoint文件只能用于python环境下的tensorflow中；其次，对于tensorflow版本也有一定的要求，版本号跨度太大的框架之间无法互相读取模型（比如旧版本中一些op被弃用的情况）。

在说本篇分享的正题之前，最后再插入一句题外话。tensorflow一直以来以其僵化、难用出名，但这么久以来仍能牢牢占据大量市场份额而没有被pytorch干掉，就是依仗其整套完整的工业化生态圈。因此虽然静态图（tf-1.x而论）如此恶心人，但tensorflow仍能和pytorch互相瓜分工业圈与学术圈。

而这篇分享主要依赖的背景，便是tensorflow为了解决模型上线这一痛点而存在的几种模型导出方式。

接下来便是本文的正题所在：

### **1、使用tf.SavedModelBuilder导出MetaGraphDef**

首先说好处：

1、使用这种方法得到的模型文件可以跨语言调用（下面给出基于java的例子）

2、使用这种方法得到的模型文件可以配合tf serving进行线上部署，减少模型上线的难度并且提升线上运行效率。这也是tensorflow官方推荐的方式。（但考虑到部署tf serving的教程需要很长的篇幅，而大部分小伙伴的实验环境中并没有这项服务，为了方便大家运行例子，因此后续的代码例子中将不会使用tf serving来进行。大家只需要知道，后面要介绍的模型导出方法依照官方的意图是推荐与tf serving配合使用的。）

3、方便模型版本控制。

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

    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input_a': a,
                "input_b": b,
                },
        outputs={'output_c': c,
                 }
    )

    builder.add_meta_graph_and_variables(
        sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'substract_demo': signature
        }
    )
    builder.save()
```

为了例子的简单易懂，本文放弃了官网的mnist例子，使用一个简单的减法作为例子。正所谓一法通万法通，看懂了这个例子后，所有基于tf-1.x且使用placeholder作为模型读入的例子都是一个道理。(使用estimator的模型的保存方法其实就是给原本模型读入数据的地方开个placeholder的口子即可，考虑到使用estimator的场合不多见，这里就不展开讲了。)

代码中的a、b、c三个op即为我们平时模型中的输入和输出，而这些数量都是可变的，你可以有多个输入，也可以有多个输出。但需要注意的是，为了将来通过导出的模型文件还原出图结构后，能够正常的运行inference过程，你需要将所有用到的输入输出op都显示命名，如上面例子中的a、b和c（或者在足够熟悉tensorflow的情况下你能搞清楚它们的得到的默认名称也行，不过为什么要为难自己。）

上面的代码例子虽说只做了简单的tensor之间的减法，但保存模型的代码却比保存为checkpoint形式要复杂的多。如果嫌这种保存方式过于繁琐，tensorflow还提供了一个精简模式的保存函数，省略了其中大部分配置，写法如下：

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

    tf.saved_model.simple_save(sess,
                               export_path,
                               inputs={'input_a': a,
                                       "input_b": b,
                                       },
                               outputs={'output_c': c,
                                        }
                               )
```

这种写法省去了SavedModelBuilder中的各种定义，最主要的就是其中的tag定义。tag的用途是可以将一个模型的多个不同的MetaGraphDef保存在一个文件下。什么时候需要多个MetaGraphDef呢？也许你想保存图形的CPU版本和GPU版本，或者你想区分训练和发布版本。这个时候tag就可以用来区分不同的MetaGraphDef，加载的时候能够根据tag来加载模型的不同计算图。

关于MetaGraphDef的定义，以及MetaGraphDef、GraphDef和Graph的区别，是另一个需要较大篇幅的话题。不太了解并且有兴趣的伙伴可以看看这篇：[Tensorflow框架实现中的“三”种图](https://zhuanlan.zhihu.com/p/31308381)。除了GraphDef对应的模型导出部分略显粗糙外，其余部分写的都挺好。而这部分内容在下面我会尽量清晰的介绍。

大部分情况下，我们都不需要将模型的多个版本保存在一起，因此`simple_save`方法往往就足以满足我们的需求。在`simple_save`方法中，系统会给一个默认的tag: “serve”，也可以用内置的`tag_constants.SERVING`这个常量。

在运行了上面的示例代码后，我们可以发现在目标文件夹下出现了一个子文件夹`1`，这是我们在代码中所写的模型版本所控制的，而在文件夹`1`下面又有一个子文件夹`variables`和一个`saved_moel.pb`文件。其中，`saved_moel.pb`中包含了该模型的MetaGraphDef图结构，而`variables`文件夹中包括了图中所有保存了的变量的参数值。

在接着讲如何使用导出的模型之前，再演示一下使用MetaGraphDef保存和恢复模型的好处之一：不受tensorflow版本限制。看下面这段代码例子，这段代码使用的是tf-1.13.1完成的：

```angular2

```

