"""
    一个CNN小练习，数据集来自https://www.kaggle.com/datasets/gpreda/chinese-mnist
    参考了https://zhuanlan.zhihu.com/p/102119808
"""

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# %%
def get_tar_jpg(id1, id2, id3):
    filename = f'./archive/data/data/input_{id1}_{id2}_{id3}.jpg'
    img = np.array(Image.open(filename))
    return img


# %% convolution object
class Filter:
    def __init__(self, width, height, depth, stride, if_paddle, learning_rate=0.01):
        self.width = width
        self.height = height
        self.depth = depth
        self.stride = stride
        self.if_paddle = if_paddle  # 这里设计的补0主要是针对stride大于1时带来边缘信息没有卷积到的情况，参考forward函数的第一个if
        self.weights = np.random.randn(height, width)
        # self.bias = np.random.randn(1)
        self.learning_rate = learning_rate

    def forward(self, picture):
        """
        对给定输入的adarray文件picture做卷积操作
        :param picture: 图片，adarray
        :return: feature map
        """
        pic_height, pic_width, pic_depth = picture.shape
        if self.if_paddle == 1:
            picture = np.pad(picture,
                             ((0, self.width - pic_width % self.width), (0, self.height - pic_height % self.height),
                              (0, 0)),
                             'constant')
        pic_height, pic_width, pic_depth = picture.shape
        out_height = (pic_height - self.height) // self.stride + 1
        out_width = (pic_width - self.width) // self.stride + 1
        feature_map = np.zeros((out_height, out_width))
        for i in range(out_height):
            for j in range(out_width):
                # relu
                feature_map[i, j] = max(np.sum(picture[i:i + self.height, j:j + self.width] * self.weights), 0)
        return feature_map


class ConvolutionLayer:
    def __int__(self):
        self.conv1 = Filter(3, 3, 3, 1, 1)
        self.conv2 = Filter(3, 3, 3, 1, 1)


class Softmax:
    def __init__(self, image):
        self.image = image

    def forward(self):
        return np.exp(self.image) / np.sum(np.exp(self.image), axis=0)


class MaxPool:
    def __init__(self, image, pool_size, stride):
        self.image = image
        self.height, self.width, self.depth = image.shape
        self.pool_size = pool_size
        self.stride = stride

    def forward(self):
        out_height = (self.height - self.pool_size) // self.stride + 1
        out_width = (self.width - self.pool_size) // self.stride + 1
        out = np.zeros((out_height, out_width, self.depth))
        for d in range(self.depth):
            for i in range(out_height):
                for j in range(out_width):
                    out[i, j] = np.max(self.image[i * self.stride:i * self.stride + self.pool_size,
                                       j * self.stride:j * self.stride + self.pool_size, d])
        return out.reshape(out_height * out_width * self.depth, 1)


class LinearLayer:
    """
    从之前的BP算法中复制过来的
    """

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.threshold = np.random.randn(output_size, 1)
        print(self.threshold)
        self.input = None
        self.output = None
        self.delta = np.zeros(np.shape(self.threshold))
        self.delta_weights = np.zeros(np.shape(self.weights))
        self.learning_rate = 0.001

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-(np.dot(input.T, self.weights) - self.threshold.T)))  # Sigmoid
        self.output = self.output.T
        return self.output

    def backward(self, dout):
        """
        反向传播
        :param dout: 上一层的输出求偏导的结果
        :return:
        """
        self.delta += dout * self.output * (1 - self.output)
        self.delta_weights += self.input * self.delta.T
        dx = np.dot(self.weights, self.delta)
        return dx

    def update(self, learning_rate):
        self.weights += self.delta_weights * learning_rate
        self.threshold += self.delta * learning_rate
        self.delta = np.zeros(np.shape(self.threshold))
        self.delta_weights = np.zeros(np.shape(self.weights))
        return self.weights, self.threshold


# %% load data
data = pd.read_csv('archive/chinese_mnist.csv')
data.drop(columns='character', inplace=True)
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['value', ]), data['value'], test_size=0.2)

# reset index
x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# %% 测试代码
pic1 = get_tar_jpg(1, 1, 1).reshape(64, 64, 1)
test_filter = Filter(3, 3, 3, 1, 1)
b = test_filter.forward(pic1).reshape(64, 64, 1)
# Image.fromarray(b.reshape(64,64)).show()
test_maxpool = MaxPool(b, 2, 2)
c = test_maxpool.forward()
d = Softmax(c).forward()
# np.count_nonzero(c == d)

# 查看data的value有几种类型
label_dict = data['value'].unique().reshape(-1, 1)
item_num = label_dict.shape[0]
label_dict = np.concatenate((np.array(range(1, item_num + 1)).reshape(-1,1),label_dict), axis=1)
full_connect = LinearLayer(1024, 14)
full_connect.forward(c)

# 检查c的非零元素个数
np.count_nonzero(c)