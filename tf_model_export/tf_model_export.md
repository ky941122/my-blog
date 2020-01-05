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
    export_path = os.path.join("./model", str(model_version))
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
```

这种写法省去了SavedModelBuilder中的各种定义，最主要的就是其中的tag定义。tag的用途是可以将一个模型的多个不同的MetaGraphDef保存在一个文件下。什么时候需要多个MetaGraphDef呢？也许你想保存图形的CPU版本和GPU版本，或者你想区分训练和发布版本。这个时候tag就可以用来区分不同的MetaGraphDef，加载的时候能够根据tag来加载模型的不同计算图。

关于MetaGraphDef的定义，以及MetaGraphDef、GraphDef和Graph的区别，是另一个需要较大篇幅的话题。不太了解并且有兴趣的伙伴可以看看这篇：[Tensorflow框架实现中的“三”种图](https://zhuanlan.zhihu.com/p/31308381)。除了GraphDef对应的模型导出部分略显粗糙外，其余部分写的都挺好。而这部分内容在下面我会尽量清晰的介绍。

大部分情况下，我们都不需要将模型的多个版本保存在一起，因此`simple_save`方法往往就足以满足我们的需求。在`simple_save`方法中，系统会给一个默认的tag: “serve”，也可以用内置的`tag_constants.SERVING`这个常量。

在运行了上面的示例代码后，我们可以发现在目标文件夹下出现了一个子文件夹`1`，这是我们在代码中所写的模型版本所控制的，而在文件夹`1`下面又有一个子文件夹`variables`和一个`saved_moel.pb`文件。其中，`saved_moel.pb`中包含了该模型的MetaGraphDef图结构，而`variables`文件夹中包括了图中所有保存了的变量的名字与参数值。

在接着讲如何使用导出的模型之前，再演示一下使用MetaGraphDef保存和恢复模型的好处之一：不受tensorflow版本限制。看下面这段代码例子，这段代码使用的是tf-1.13.1完成的：

```angular2
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
   
    model_version = 1
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
```

在这段代码里，我们使用到了tf.keras作为网络的一部分。这在做实验的过程中是很常见的，keras模块中的高级抽象层结构能让我们更加轻松快速的搭建模型。但是，假如你现在训练好了一个不错的模型准备上线，但是线上机器的tf版本却是古老的tf-1.2，这时候若是采用类似实验过程中的重新定义一遍网络结构再使用tf.train.Saver来载入checkpoint的方式就行不通了。
因为在tf-1.2中还没有引入tf.keras这个模块，因此代码在`temp2 = tf.keras.layers.Dense(units=16)(temp1)`这一行的时候就会报错找不到tf.keras模块，然后退出。现在我们来看一下使用MetaGraphDef载入上面示例中保存的模型的方法：

```angular2
#coding=utf-8
import tensorflow as tf
import os
import numpy as np

with tf.Session(graph=tf.Graph()) as sess:
    model_version = 1
    export_path = os.path.join("./model", str(model_version))

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)

    input_a = sess.graph.get_operation_by_name('a').outputs[0]
    input_b = sess.graph.get_operation_by_name('b').outputs[0]
    output_c = sess.graph.get_operation_by_name('c').outputs[0]

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
```

上面这段代码即使在tf-1.2下也能运行，因为虽然在1.2版本中并未引入keras模块，但该模块下的抽象网络层也是通过底层api来实现的，因此使用MetaGraphDef可以直接载入模型图结构。

上面说到使用checkpoint无法在tf-1.2下载入该模型，其实是不够准确的，要想使用checkpoint，有两个方法：

第一个方法，是使用`tf.train.Saver`保存checkpoint后的meta文件进行载入。以meta结尾的checkpoint文件中保存了和使用`tf.SavedModelBuilder`获得的同样的MetaGraphDef数据，因此同样可以直接还原出模型图结构。具体用法如下：

```angular2
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
```

```angular2
sess = tf.Session()
with sess.as_default():
    saver = tf.train.import_meta_graph('./model/ckpt/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/ckpt'))

    input_a = sess.graph.get_operation_by_name('a').outputs[0]
    input_b = sess.graph.get_operation_by_name('b').outputs[0]
    output_c = sess.graph.get_operation_by_name('c').outputs[0]

    a = np.array(
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    )
    b = np.array(
        [
            [6, 7, 8],
            [7, 8, 9]
        ]
    )
    feed_dict = {
        input_a: a,
        input_b: b
    }

    c = sess.run(output_c, feed_dict=feed_dict)
    print(c)
```

上面的第二段代码同样可以在tf-1.2下运行，因为`.meta`文件中保存的仍是同样的MetaGraphDef数据。至于checkpoint与MetaGraphDef的区别，将在本文的最后部分进行说明。

刚刚说了，使用checkpoint在tf-1.2下还原keras模块网络层有两个方法。一个是上面介绍了的同样使用checkpoint中保存的MetaGraphDef信息进行图结构加载。另一个，还是大家最常接触到的，重新定义网络结构，并使用`tf.train.Saver`载入参数。但是为了避免tf.keras模块不存在的问题，这时候就需要我们自己仿造该模块下的网络层实现方式，使用底层api进行模型构造，并且不要忘了，还得使用和keras模块中相同的名空间才行，不然网络参数仍然无法读取成功。
关于这个方法，本文就不多做演示了。因为这并不是一个在真正实践中推荐的做法。像上面的只有一个全连接层的小例子还好，但若涉及到了复杂网络，就是一个大工程了。

在本节的最后，以java为例，讲一下tensorflow中的MetaGraphDef如何进行跨语言调用。这其实是一个挺重要的技巧，因为大型项目的后端代码往往都不会是python为主的，这时候如果要在中间插入一个python实现的模型其实是比较尴尬的，若是能使用和整个架构相同的语言进行模型调用，那么何乐而不为呢。

首先，新建一个maven项目，并在`pom.xml`配置文件中添加tensorflow依赖：

```angular2
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>1.13.1</version>
</dependency>
```

简单起见，这里使用最开始的小例子中保存的减法模型，将其生成的`1`文件夹下的`variables`子文件夹（其实可以忽略这个文件夹，因为未使用任何变量的原因，这是个空文件夹，但为了保证结构的完整性，还是把它带上）和`saved_model.pb`文件一起，移动到刚刚创建的java项目的resources文件夹下。然后运行下面这段代码：

```angular2
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.SavedModelBundle;

public class Main {
    public static void main(String[] args){
        SavedModelBundle model_bundle = SavedModelBundle.load("F:\\work\\tf_test\\src\\main\\resources", "serve");
        Session model_session = model_bundle.session();

        float[] a = {1, 2, 3, 4, 5};
        float[] b = {4, 5, 6, 7, 8};

        Tensor input_a = Tensor.create(a);
        Tensor input_b = Tensor.create(b);
        Tensor<Float> result = model_session.runner()
                .feed("a", input_a)
                .feed("b", input_b)
                .fetch("c")
                .run()
                .get(0)
                .expect(Float.class);

        long[] rshape = result.shape();
        int batchSize = (int) rshape[0];
        float[] output_c = new float[batchSize];
        output_c = result.copyTo(output_c);

        for (float x: output_c) {
            System.out.print(x);
            System.out.print("\t");
        }
    }
}
```

载入过程其实与python中类似，先通过MetaGraphDef还原出模型图结构，再使用session运行要输出的op，对于placeholder的feed方式也与python中类似。因此，当inference过程中对于输入数据的预处理不涉及特别多的科学计算的话，比如对于大部分NLP模型，对于输入文本的预处理就只有分词与词表映射了，这种情况下，使用java或c++进行模型部署，就是一种唾手可得的模型提速方法。

### **2、使用tf.graph_util.convert_variables_to_constants导出GraphDef**

上面提到过，在tensorflow中，图信息包含三种，Graph、MetaGraphDef和GraphDef。GraphDef是MetaGraphDef中的一个核心内容（关于三者的区别和联系，还是推荐不熟悉的小伙伴看一下这个专栏：[Tensorflow框架实现中的“三”种图](https://zhuanlan.zhihu.com/p/31308381)，里面包括了本文中关于图的所有前置知识）。保存和使用GraphDef并不能与官方推荐的tf serving相配合，但却是移动端或其他需要轻量模型的场景下的首选。

还是首先说说好处：

1、使用这种方法得到的模型文件可以跨语言调用（下面给出基于java的例子）

2、GraphDef中将所有变量常数化了，显著减小了模型的大小，适合移动端部署。

但是GraphDef却有一个很明显的缺点：使用GraphDef重建的图结构无法重新用于训练，这是变量常数化后必须支付的一个代价。

下面上例子，首先看下保存GraphDef的例子：

```angular2

```

