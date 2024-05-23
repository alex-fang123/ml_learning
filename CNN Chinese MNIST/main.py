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


# %% convolution
class filter:
    def __init__(self, width, height, depth, stride, if_paddle, learning_rate=0.01):
        self.width = width
        self.height = height
        self.depth = depth
        self.stride = stride
        self.if_paddle = if_paddle  # 这里设计的补0主要是针对stride大于1时带来边缘信息没有卷积到的情况，参考forward函数的第一个if
        self.weights = np.random.randn(height, width)
        self.bias = np.random.randn(1)
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
                             ((0, self.height - pic_height % self.height), (0, self.width - pic_width % self.width)),
                             'constant')
        pic_height, pic_width, pic_depth = picture.shape
        out_height = (pic_height - self.height) // self.stride + 1
        out_width = (pic_width - self.width) // self.stride + 1
        feature_map = np.zeros((out_height, out_width))
        for i in range(out_height):
            for j in range(out_width):
                feature_map[i, j] = np.sum(picture[i:i + self.height, j:j + self.width] * self.weights) + self.bias
        return feature_map


class softmax:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1)
        self.learning_rate = learning_rate

    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

# %% load data
data = pd.read_csv('archive/chinese_mnist.csv')
data.drop(columns='character', inplace=True)
# pic1 = get_tar_jpg(1, 1, 1)
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['value', ]), data['value'], test_size=0.2)

# reset index
x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
