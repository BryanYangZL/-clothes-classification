# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
import matplotlib

data_dir = 'data/'

# 下载中文支持字体。后面画图需要
zhfont = matplotlib.font_manager.FontProperties(fname= data_dir + 'SimHei-windows.ttf')


# 解析解压得到四个训练的数据
def read_data():
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    # 我在当前的目录下创建文件夹，里面放入上面的四个压缩文件
    paths = []
    for i in range(len(files)):
        paths.append(data_dir + files[i])

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

# 分别得到训练数据集和测试数据集
(train_images, train_labels), (test_images, test_labels) = read_data()

class_names = ['短袖圆领T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋','包', '短靴']

# 训练图像缩放255，在0 和 1 的范围
train_images = train_images / 255.0

# 测试图像缩放
test_images = test_images / 255.0

# 保存画布的图形，宽度为 10 ， 长度为10
plt.figure(figsize=(10, 10))

# 显示训练集的 25 张图像
for i in range(25):
    # 创建分布 5 * 5 个图形
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 显示照片，以cm 为单位。
    plt.imshow(train_images[i], cmap=plt.cm.binary)

    # 此处就引用到上面的中文字体，显示指定中文，对应下方的图片意思，以证明是否正确
    plt.xlabel(class_names[train_labels[i]], fontproperties=zhfont)

# 建立模型
def build_model():
    # 线性叠加
    model = tf.keras.models.Sequential()
    # 改变平缓输入
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # 第一层紧密连接128神经元
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # 第二层分10 个类别
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    return model

# 编译模型
model = build_model()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5)

# 评估模型（主要是测试数据集）
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('测试损失：%f 测试准确率: %f' % (test_loss, test_acc))

predictions = model.predict(test_images)
# 提取20个数据集，进行预测判断是否正确
for i in range(25):
    pre = class_names[np.argmax(predictions[i])]
    tar = class_names[test_labels[i]]
    print("预测：%s   实际：%s" % (pre, tar))




