"""
    一个CNN小练习，数据集来自https://www.kaggle.com/datasets/gpreda/chinese-mnist
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


#%% convolution
class filter:
    def __init__(self, width, height, depth, stride, learning_rate=0.01):
        self.width = width
        self.height = height
        self.depth = depth
        self.stride = stride
        self.weights = np.random.randn(height, width)
        self.bias = np.random.randn(1)

    def forward(self, picture):
        """
        对给定输入的adarray文件picture做卷积操作
        :param picture: 图片，adarray
        :return: feature map
        """
        pic_height, pic_width, pic_depth = picture.shape
        out_height = (pic_height - self.height) // self.stride + 1
        out_width = (pic_width - self.width) // self.stride + 1
        feature_map = np.zeros((out_height, out_width))
        for i in range(out_height):
            for j in range(out_width):
                feature_map[i, j] = np.sum(picture[i:i + self.height, j:j + self.width] * self.weights) + self.bias
        return feature_map


# %% load data
data = pd.read_csv('archive/chinese_mnist.csv')
data.drop(columns='character', inplace=True)
# pic1 = get_tar_jpg(1, 1, 1)
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['value',]),data['value'], test_size=0.2)

# reset index
x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)