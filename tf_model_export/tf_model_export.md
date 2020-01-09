# Tensorflow模型导出与跨语言导入

相信使用过tensorflow的人都接触过它的模型保存功能，这在我们日常实验中经常用到的：使用预训练模型、测试已有模型以及继续训练现有模型等场景中都是必备的技巧。在工作学习中的大多数时候我们使用的保存模型的api都是：

```angular2
saver = tf.train.Saver()
saver.save()
```

运行之后就可以得到四个文件：checkpoint、model.ckpt.data、model.ckpt.index、model.ckpt.meta，分别保存了已导出的checkpoint文件列表信息、网络权值、变量名称索引信息以及图结构信息。

使用这种方式进行模型的保存与读取对于实验过程来说完全足够了，但若是我们通过实验得到了一个不错的模型，现在想要将它部署上线，上面说的checkpoint文件就显得有些不足了（本文后续都将上述的四个文件统一称作checkpoint文件，而不是特指名为`checkpoint`的那个特定文件）。这里主要存在着这样几个缺点：首先最明显的是，checkpoint文件只能用于python语言下的tensorflow框架中；其次，对于tensorflow版本也有一定的要求，版本号跨度太大的框架之间想要互相读取模型相对来说比较困难（比如旧版本中一些api被弃用的情况）。

在说本篇分享的正题之前，最后再插入一句题外话。tensorflow一直以来以其僵化、难用出名，但这么久以来仍能牢牢占据大量市场份额而没有被pytorch干掉，就是依仗其整套完整的工业化生态圈。因此虽然静态图（tf-1.x而论）如此恶心人，但tensorflow仍能和pytorch互相瓜分工业圈与学术圈。

而这篇分享主要依赖的背景，便是tensorflow为了解决深度学习模型上线这一痛点而存在的几种模型导出方式。

接下来便是本文的正题所在：

### **1、使用tf.saved_model导出MetaGraphDef**

这种模型保存方式也是我一直以来最优先使用的模型导出上线的方式，因此放在最前面介绍。首先说说这种模型保存方式的好处：

1、使用这种方法得到的模型文件可以跨语言调用（下面给出基于java的例子）

2、使用这种方法得到的模型文件可以配合tf serving进行线上部署，减少模型上线的难度并且提升线上运行效率。这也是tensorflow官方推荐的方式。（但考虑到配置tf serving的教程需要很长的篇幅，而大部分小伙伴的实验环境中并没有配置这项服务，为了方便大家运行例子，因此本文中后续的代码例子中将不会使用tf serving来进行。大家只需要知道，本节中要介绍的模型导出方法依照官方的意图是推荐与tf serving配合使用的。）

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

    builder = tf.saved_model.builder.saved_model(export_path)

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

为了例子的简单易懂，本文放弃了官网的mnist例子，使用一个简单的减法作为例子。正所谓一法通万法通，看懂了这个例子后，所有基于tf-1.x且使用placeholder作为模型输入的例子都是一个道理。(使用estimator的模型的保存方法其实就是给原本模型读入数据的地方开个placeholder的口子即可，考虑到使用estimator的场合不多见，这里就不展开讲了。)

代码中的a、b、c三个op即为我们平时模型中的输入和输出，它们的数量都是可变的，你可以有多个输入，也可以有多个输出。但需要注意的是，为了将来通过导出的模型文件还原出图结构后，能够正常的运行inference过程，你需要将所有会用到的输入输出op都显示命名，如上面例子中的a、b和c。因为我们稍后需要使用这些名称来重新获得这些op。（或者在足够熟悉tensorflow的情况下，你能搞清楚它们会得到的默认名称也行，不过为什么要为难自己。）

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

这种写法省去了saved_model中的各种定义，最主要的就是其中的tag定义。tag的用途是可以将一个模型的多个不同的MetaGraphDef保存在一个文件下。什么时候需要多个MetaGraphDef呢？也许你想保存图形的CPU版本和GPU版本，或者你想区分训练和发布版本。这个时候tag就可以用来区分不同的MetaGraphDef，加载的时候能够根据tag来加载模型的不同计算图。

关于MetaGraphDef的定义，以及MetaGraphDef、GraphDef和Graph的区别，是另一个需要较大篇幅的话题。不太了解并且有兴趣的伙伴可以看看这篇专栏：[Tensorflow框架实现中的“三”种图](https://zhuanlan.zhihu.com/p/31308381)。（这篇专栏中除了GraphDef对应的模型导出部分略显粗糙外，其余部分写的都挺好。而这部分内容在下面我会尽量清晰的介绍。）

大部分情况下，我们都不需要将模型的多个版本保存在一起，因此`tf.saved_model.simple_save`方法往往就足以满足我们的需求。在`tf.saved_model.simple_save`方法中，系统会给一个默认的tag: “serve”，也等价于tensorflow内置的`tag_constants.SERVING`这个常量。

在运行了上面的示例代码后，我们可以发现在目标文件夹下出现了一个子文件夹`1`，这是我们在代码中所写的模型版本所控制的，而在文件夹`1`下面又有一个子文件夹`variables`和一个`saved_moel.pb`文件。其中，`saved_moel.pb`中包含了该模型的MetaGraphDef信息，而`variables`文件夹中保存了图中所有被保存的变量的索引信息与参数值。

在接着讲如何载入刚才导出的模型之前，再演示一下使用saved_model保存和恢复模型的好处之一：不受tensorflow版本限制。看下面这段代码例子，这段代码使用的是tf-1.13.1完成的：

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

在这段代码里，我们使用到了tf.keras作为网络的一部分。这在做实验的过程中是很常见的，keras模块中的高级抽象层结构能让我们更加轻松快速的搭建模型。但是，假如你现在训练好了一个不错的模型准备上线，但是线上机器的tf版本却是古老的tf-1.2，这时候若是采用我们平时在实验过程中经常使用的：重新定义一遍网络结构再使用tf.train.Saver来载入checkpoint的方式就行不通了。
因为在tf-1.2中还没有引入tf.keras这个模块，因此在重新定义网络的过程中，代码在`temp2 = tf.keras.layers.Dense(units=16)(temp1)`这一行的时候就会报错找不到tf.keras模块，然后退出。现在我们来看一下使用saved_model载入上面示例中保存的模型的方法：

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

上面这段代码即使在tf-1.2下也能运行，因为虽然在1.2版本中并未引入keras模块，但该模块下的抽象网络层也是通过底层api来实现的，因此使用saved_model保存的MetaGraphDef可以直接载入模型图结构。

关于使用saved_model进行模型恢复唯一需要注意的一点就是：在`tf.saved_model.loader.load`函数中的第二个参数，我们需要传入模型保存时输入的tag，若是使用的简单保存方法，那tag就是tf默认传入的“serve”或者预定义常量`tag_constants.SERVING`。

上面说到使用checkpoint无法在tf-1.2下载入该模型，其实是不够准确的，要想使用checkpoint，有两个方法：

第一个方法，是使用`tf.train.Saver`保存checkpoint后的meta文件进行载入。以meta结尾的checkpoint文件中保存了和使用`tf.saved_model`获得的`saved_model.pb`中同样的MetaGraphDef数据，因此同样可以直接还原出模型图结构。具体用法如下：

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

刚刚说了，使用checkpoint在tf-1.2下还原keras模块网络层有两个方法。一个是上面介绍了的同样使用checkpoint中保存的MetaGraphDef信息进行图结构加载。另一个，还是大家最常接触到的，重新定义网络结构，并使用`tf.train.Saver`载入权值。但是为了避免tf.keras模块不存在的问题，这时候就需要我们自己仿造该模块下的网络层实现方式，使用底层api进行模型构造，并且不要忘了，还得使用和keras模块中相同的命名空间才行，不然网络权值仍然无法读取成功。
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

简单起见，这里使用最开始的小例子中保存的减法模型，将其生成的`1`文件夹下的`variables`子文件夹（其实可以忽略这个文件夹，因为减法例子中未使用任何变量的原因，这是个空文件夹，但为了保证结构的完整性，还是把它带上）和`saved_model.pb`文件一起，移动到刚刚创建的java项目的resources文件夹下。然后运行下面这段代码：

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

载入过程其实与python中类似，先通过MetaGraphDef还原出模型图结构，再使用session运行要输出的op，对于placeholder的feed方式也与python中类似。因此，当模型在inference过程中对于输入数据的预处理不涉及特别多的科学计算的话，比如对于大部分NLP模型，对于输入文本的预处理就只有分词与词表映射了，这种情况下，使用java或c++进行模型部署，就是一种唾手可得的模型提速方法。

### **2、使用tf.graph_util.convert_variables_to_constants导出GraphDef**

上面提到过，在tensorflow中，图信息包含三种，Graph、MetaGraphDef和GraphDef。GraphDef是MetaGraphDef中的一个核心部分（关于三者的区别和联系，还是推荐不熟悉的小伙伴看一下这个专栏：[Tensorflow框架实现中的“三”种图](https://zhuanlan.zhihu.com/p/31308381)，里面包括了本文中关于图的所有前置知识）。保存和使用GraphDef并不能与官方推荐的tf serving相配合，但却是移动端或其他需要轻量模型的场景下的首选。

还是首先说说好处：

1、使用这种方法得到的模型文件可以跨语言调用（下面给出基于java的例子）

2、GraphDef中将所有变量常数化了，显著减小了模型的大小，适合移动端部署。

但是GraphDef却有一个很明显的缺点：使用GraphDef重建的图结构无法重新用于训练，这是变量常数化后必须支付的一个代价。

下面上例子，首先看下保存GraphDef的例子：

```angular2
#coding=utf-8
import tensorflow as tf

sess = tf.Session()
with sess.as_default():

    a = tf.placeholder(shape=[None], dtype=tf.float32, name="a")
    b = tf.placeholder(shape=[None], dtype=tf.float32, name="b")
    c = tf.subtract(a, b, name="c")

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["c"])
    with tf.gfile.GFile("./model/graphDef/model.pb", "wb") as model_f:
        model_f.write(output_graph_def.SerializeToString())
```

代码十分清晰明了，通过`sess.graph_def`获得当前session所在图的GraphDef，然后调用`tf.graph_util.convert_variables_to_constants`将变量转换为常数，最后序列化后写入文件。这里唯一要注意的一点是，不像保存MetaGraphDef的时候，tensorflow会自动将整张图保存下来，保存GraphDef需要我们手动输入对于现有模型我们想要获得的输出，tensorflow根据这个输出，倒推出它所依赖的子图，然后将这个子图保存下来。
上面的小例子里，我们想要得到的输出是c节点，因此在`tf.graph_util.convert_variables_to_constants`函数的`output_node_names`参数中传入c，注意要传节点的名称，且要使用列表的形式，这是因为如果你的模型有多个输出，那就可以一并列出在列表中。大家可以试试，如果在`output_node_names`参数里传入的是节点a，那么这个GraphDef里就只会保存一个节点a，这是因为想要获得节点a，并不需要依赖于其他别的节点。

除了上面这个导出为.pb格式的方式外，保存图的GraphDef还有另外一种导出为.pbtxt的方式。

```angular2
#coding=utf-8
import tensorflow as tf

sess = tf.Session()
with sess.as_default():

    a = tf.placeholder(shape=[None], dtype=tf.float32, name="a")
    b = tf.placeholder(shape=[None], dtype=tf.float32, name="b")
    c = tf.subtract(a, b, name="c")

    tf.train.write_graph(sess.graph_def, "./model/graphDef", "model.pbtxt", as_text=True)
```

这里值得一提的是，本文一直出现的.pb文件扩展名，其实是Protocol Buffer（protobuf）的简写，这是谷歌开源的一种数据存储语言，tensorflow中保存模型使用的都是这种存储语言，无论是.ckpt的checkpoint，还是.pb和.pbtxt的MetaGraphDef或GraphDef。关于Protocol Buffer，本文不会展开，有兴趣了解的小伙伴可以自行查阅资料。唯一需要说明的是，.pb与.pbtxt是protobuf的两种储存形式，前者是以二进制形式存储数据，因此获得的文件在大小上会有显著的降低；后者是以文本形式存储，大小上相对前者会有增加，但好处是存储的文件是人类可阅读的形式，在某些场合下有利于进行debug。
我们可以打开使用`tf.train.write_graph`api保存的`model.pbtxt`文件进行查看。（需要注意的是，以何种形式保存protobuf并不是根据你所声明的文件名后缀来区分的，而是根据`tf.train.write_graph`函数中的`as_text`参数来决定的，这个参数默认为True，也就是以文本形式保存。但是一个清晰明了的文件名后缀有时候能避免很多不必要的麻烦）。

```angular2
node {
  name: "a"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
      }
    }
  }
}
```

这是`model.pbtxt`文件中对于我们上面定义的图结构中的a节点所保存的信息，可以看到对于该节点的每条信息我们都能轻松的读懂。但由于保存GraphDef的一个最主要的目的就是其文件大小优势，因此会使用文本形式存储的场合其实不多。

接下来展示一下如何读取上面保存的`model.pb`与`model.pbtxt`文件。

首先是读取二进制形式的文件：

```angular2
#coding=utf-8
import tensorflow as tf
import numpy as np

model_filename = "./model/graphDef/model.pb"
with tf.gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
    input_a = sess.graph.get_tensor_by_name('a:0')
    input_b = sess.graph.get_tensor_by_name('b:0')
    output_c = sess.graph.get_tensor_by_name('c:0')

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
```

然后是读取文本形式的文件：

```angular2
#coding=utf-8
import tensorflow as tf
import numpy as np
from google.protobuf import text_format

model_filename = "./model/3/model.pbtxt"
with tf.gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    file_content = f.read()
    text_format.Merge(file_content, graph_def)

tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
    input_a = sess.graph.get_tensor_by_name('a:0')
    input_b = sess.graph.get_tensor_by_name('b:0')
    output_c = sess.graph.get_tensor_by_name('c:0')

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
```

上面两段导入模型的代码也很简洁，只有一个需要注意的地方，在使用`tf.import_graph_def`函数将从文件中获取的GraphDef信息导入图中时，函数有一个参数`name`，默认值是"import"。这个参数的作用是给导入的图中所有节点的名称加上一个前缀，也就是用一个名空间方便的区分出导入的节点。
如果不传入这个参数，那么图中所有节点的名字都会获得一个前缀"import/"，在不注意的情况下很容易出现找不到对应名字节点的错误。所以方便起见，我一般会传入`name=""`来消除这个额外添加的名空间。

这里额外插入一句，由于上面介绍的所有方法，在载入保存好的图后，都需要通过op的名称来获取到图中定义好的节点，因此有时候你会需要查看图里有哪些节点名称。我常用的方法是通过`tf.get_default_graph().as_graph_def().node`来获取所有节点的集和，并对集和中的每个元素查看其name属性，这就是所有节点的名称。实现类似功能的方法还有很多，大家可以选用自己顺手的。

上面提到过，使用MetaGraphDef可以恢复出一个可以重新训练的模型，但是使用GraphDef恢复出的模型却不可以。这是因为GraphDef中将所有的变量都固化成为了常数。这里我们简单印证下这个结论，我们可以在通过GraphDef恢复出的图结构下，调用`tf.global_variables`函数，打印出结果会发现返回的是一个空集合。但是在通过MetaGraphDef恢复出的图结构下调用该函数，就能看到图中所定义的所有变量。（使用这个减法例子肯定看不出区别，因为例子里根本没定义变量，请务必至少定义一个变量再来印证这一点。）

这一节最后照例贴一下在java中载入并使用保存好的GraphDef数据。由于我并未通过java调用过以文本形式保存的GraphDef文件，这里就不多做展示以免发生误导，有这方面需要的可以自行查阅资料。

首先还是像上一节一样在maven中添加tensorflow依赖，然后把刚刚导出的包含GraphDef信息的model.pb文件放到项目的resources文件夹下。

```angular2
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import java.nio.file.Path;
import java.nio.file.Files;
import java.io.IOException;
import java.nio.file.Paths;

public class Main {
    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: "
                    + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    public static void main(String[] args){
        byte[] graphDef = readAllBytesOrExit(Paths.get("F:\\work\\tf_test\\src\\main\\resources", "model.pb"));
        Graph g = new Graph();
        g.importGraphDef(graphDef);
        Session sess = new Session(g);
        float[] a = {1, 2, 3, 4, 5};
        float[] b = {4, 5, 6, 7, 8};

        Tensor input_a = Tensor.create(a);
        Tensor input_b = Tensor.create(b);
        Tensor<Float> result = sess.runner()
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

与第一节的java例子中tensorflow可以自动读取MetaGraphDef模型文件略有不同的是，此时我们需要一个辅助函数来帮助我们从GraphDef文件中读取出内容并存在一个byte数组里，然后通过这个byte数组来恢复我们保存的图结构。其余部分都与python中调用十分一致了。

### **3、使用tf.train.Saver保存checkpoint**

关于使用checkpoint进行的模型保存与载入，相信所有使用过tensorflow的人都有所了解，否则恐怕连简单的模型测试都无法完成。因此，这一节只简单说下checkpoint与使用tf.saved_model导出的MetaGraphDef的区别。

如果你仔细观察第一节中我们通过tf.saved_model导出的模型文件夹中的文件，你会发现，在`variables`文件夹下，存在两个文件`variables.data`和`variables.index`（当然，这里指的是包括一个全连接层的例子。简单的减法模型例子导出的`variables`文件夹是空的）。你或许会觉得这两个文件看起来很熟悉，这是因为这两个文件的命名方式与使用tf.train.Saver导出的checkpoint中的`model.data`与`model.index`十分类似。事实上，这些文件里保存的东西也是差不多的，分别都是变量的索引信息以及网络权值。
再来看看tf.saved_model导出的与`variables`文件夹位于同一级的`saved_model.pb`，上面不止一次提到了，里面保存的是图的MetaGraphDef，而checkpoint中的`model.meta`文件里，保存的也是MetaGraphDef。这么看来，似乎除了保存了checkpoint文件列表信息的`checkpoint`文件外（这句话看起来似乎有点绕口，前一个checkpoint指这一类文件，后一个checkpoint特指名字为`checkpoint`的那个checkpoint文件），每次我们使用tf.train.Saver保存checkpoint都会多出的三个文件都能与tf.saved_model保存的文件相对应上，那他们之间到底有啥区别呢？

在我看来，区别主要可以分为两点：

首先，checkpoint文件主要是用于实验过程中使用的，因此是图结构与网络权重互相分离的几个文件。你可以在载入权重时，通过重新定义网络结构，只载入部分网络权值，然后在后面接上与下游任务适配的其他网络结构继续训练。这在图像处理与近期的自然语言处理中频繁使用的pretrain-finetune两阶段训练中十分常见。
也可以不重新定义网络，而是直接通过`model.meta`文件直接恢复整个图结构，这在第一节中已经举过例子。但是如果你仔细观察第一节中的例子就会发现。这种方式的模型载入其实也是分成两步的。首先通过`tf.train.import_meta_graph`使用MetaGraphDef文件恢复网络结构，再通过`saver.restore`来载入网络权值。如果不载入网络权值的话，网络中的参数仍然是随机初始化的。这其实就是把手动定义网络结构这个步骤给简化了而已，仍然充分体现了结构与权值分离的思想。

但saved_model导出的pb文件却不同，你只需要使用一条`tf.saved_model.loader.load`命令就可以同时加载图结构与网络权值，这是因为tensorflow中的这种模型导出载入方法就是为了模型部署发布而设计的，因此不像checkpoint那样为了服务于实验过程而相对灵活，虽然saved_model导出的模型文件仍然分成了三个文件，但更像是一个整体。

其次，也是我觉得最重要的一个区别，checkpoint文件只能用于python语言下使用，无法使用其他语言进行调用，更无法配合tf serving进行模型部署。同样的一个深度网络模型，在java中进行inference的速度远远快于在python中进行，而使用tf serving部署的模型进行inference的速度还要更快一些。

**以上就是tensorflow中模型导出导入的几个方法与例子。**
