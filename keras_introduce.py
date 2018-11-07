import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import  Dense,Flatten,Conv2D,MaxPooling2D
from keras import backend as K
num_classes=10
img_rows,img_cols=28,28
#通过Keras封装好的API加载MNIST数据，其中trainX就是一个60000X28X28数组，trainY是每一张图片对应的数字
(trainX,trainY),(testX,testY)=mnist.load_data()
#因为不同的底层（tensorflow 或者 MXnet）对输进去的要求不一样，所以这里需要根据图像编码的格式要求来设置输进层的格式。
if K.image_data_format()=="channels_first":
    trainX=trainX.reshape(trainX.shape[0],1,img_rows,img_cols)
else:
    trainX=trainX.reshape(trainX.shape[0],img_rows,img_cols,1)
    testX=testX.reshape(testX.shape[0],img_rows,img_cols,1)
    input_shape=(img_rows,img_cols,1)

#将图像像素转化为0到1之间的实数
trainX=trainX.astype("float32")
testX=testX.astype("float32")
trainX/=255.0
testX/=255.0

#将标准答案转化为需要的格式（one-hot 编码）
trainX=keras.utils.to_categorical(trainY,num_classes)
testY=keras.utils.to_categorical(testY,num_classes)

#使用keras API定义模型
model=Sequential()
#一层深度为32，过滤器大学为5x5的卷积层
model.add(Conv2D(32,kernel_size=(5,5),activation="relu",input_shape=input_shape))
#一层过滤器的大小为2X2的最大池化层
model.add(MaxPooling2D(pool_size=(2,2)))
#将卷积层的输出位置拉直后作为下面全连接层的输进
model.add(Flatten())
#全连接层，有500个节点
model.add(Dense(500,activation="relu"))
#全连接层，得到最后的输出
model.add(Dense(num_classes,activation="softmax"))

#定义损失函数，优化函数和评测函数的方法
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(),metrics=["accuracy"])

#类似TFlearn 中的训练过程，给出训练 数据，batch大小，训练轮数和验证数据。keras可以自动完成模型的训练过程
model.fit(trainX,trainY,
          batch_size=128,
          epochs=20,
          validation_data=(testX,testY))
#测试数据上计算的准确率
score=model.evaluate(testX,testY)
print("test loss:",score[0])
print("Test accuracy:",score[1])


