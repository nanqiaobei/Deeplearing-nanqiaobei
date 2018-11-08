import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#j将tensorflow日志信息输进到屏幕。
tf.logging.set_verbosity(tf.logging.INFO)
mnist=input_data.read_data_sets("路径",one_hot=False)

#指定神经网络的输进层，这里指定的输进都会拼接在一起作为整个神经网络的输进

feature_columns=[tf.feature_column.numeric_column("image",shape=[784])]

#t通过tensorflow提供的封装好的Estimator定义神经网络模型，feature_columns参数给出了神经网络输进层需要的输进数据，hidden_units
#参数给出了神经网络的结构，注意DNNClassifier只能定义多层全连接层神经网络，而hidden_units列表中给出了每一层隐藏层的节点个数，
#n_classes给出了总共类目的数量，optimizer给出了使用的优化函数，Estimator会将模型训练过程中的loss变化以及一些其他指标保存到model_dir
#目录下，通过tensorflow可以可视化这些指标的变化过程，
estimator=tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[500],n_classes=10,optimizer=tf.train.AdamOptimizer(),model_dir="D:/PycharmProject/demo1/save_log_dir")

#定义数据输进，这里x中需要给出所有的输进数据，因为上面feature_columns只定义了一组输进，所以这里只需要指定一个就好，如果feature_columns中定义了多个，
#那么这里也需要对每一个指定的输进的提供数据，y中需要提供每一个x对应的正确答案，这里要求分类的结果是一个正整数，num_epochs指定了数据循环使用的轮数。
#比如在测试可以将这个参数指定为1，batch_sizez指定了一个batch的大小。shuffle指定了是否需要对数据进行随机打乱

train_input_fn=tf.estimator.inputs.numpy_input_fn(
    x={"image":mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True

)

#训练模型，注意这里没有指定损失函数，通过DNNClassifier定义的模型会使用交叉嫡作为损失函数
estimator.train(input_fn=train_input_fn,steps=10000)

#定义测试的数据输进，指定的形式额训练是的数据输进基本一致
test_input_fn=tf.estimator.inputs.numpy_input_fn(
    x={"image":minst.test.images},
    y=mnist.test.labels.astype(np.int32),
    num_epochs=1,
    batch_size=128,
    shuffle=False
)

#通过evaluate评测训练好的模型效果
accuracy_score=estimator.evaluate(input_fn=test_input_fn)["accuracy"]
print("/nTest accuracy:%g %%"% (accuracy_score*100))