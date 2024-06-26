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


# %% object
class Filter:
    def __init__(self, width, height, depth, stride, if_paddle, learning_rate=0.01):

        self.width = width
        self.height = height
        self.depth = depth
        self.stride = stride
        self.if_paddle = if_paddle  # 这里设计的补0主要是针对stride大于1时带来边缘信息没有卷积到的情况，参考forward函数的第一个if
        self.weights = np.random.randn(height, width, depth)
        self.dw = np.zeros_like(self.weights)
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
        self.picture = picture
        pic_height, pic_width, pic_depth = picture.shape
        out_height = (pic_height - self.height) // self.stride + 1
        out_width = (pic_width - self.width) // self.stride + 1
        feature_map = np.zeros((out_height, out_width))
        for i in range(out_height):
            for j in range(out_width):
                # relu
                feature_map[i, j] = max(np.sum(picture[i * self.stride:i * self.stride + self.height,
                                               j * self.stride:j * self.stride + self.width, :] * self.weights), 0)
        return feature_map

    def backward(self, dout):
        """
        反向传播
        :param dout: 上一层的输出求偏导的结果
        :return:
        """
        self.dout = dout[:,:,np.newaxis]
        for i in range(0, self.height, self.stride):
            for j in range(0, self.width, self.stride):
                self.dw += self.dout[i:i + 1, j:j + 1, :] * self.picture[
                                                                i * self.stride:i * self.stride + 1,
                                                                j * self.stride:j * self.stride + 1, :]

    def update(self):
        self.weights += self.learning_rate * self.dw
        self.dw = np.zeros_like(self.weights)
        return self.weights


class ConvolutionLayer:
    def __init__(self, width, height, depth, stride, if_paddle):
        self.width, self.height, self.depth, self.stride, self.if_paddle = width, height, depth, stride, if_paddle
        self.conv1 = Filter(self.width, self.height, self.depth, self.stride, self.if_paddle)
        self.conv2 = Filter(self.width, self.height, self.depth, self.stride, self.if_paddle)
        self.conv3 = Filter(self.width, self.height, self.depth, self.stride, self.if_paddle)
        self.conv4 = Filter(self.width, self.height, self.depth, self.stride, self.if_paddle)

    def forward(self, picture):
        out1 = self.conv1.forward(picture)
        out2 = self.conv2.forward(picture)
        out3 = self.conv3.forward(picture)
        out4 = self.conv4.forward(picture)
        return np.array((out1, out2, out3, out4))

    def backward(self, dout):
        self.conv1.backward(dout[0])
        self.conv2.backward(dout[1])
        self.conv3.backward(dout[2])
        self.conv4.backward(dout[3])

    def update(self):
        self.conv1.update()
        self.conv2.update()
        self.conv3.update()
        self.conv4.update()


class Softmax:
    def __init__(self, image):
        self.image = image

    def forward(self):
        temp = np.sum(np.sum(np.exp(self.image), axis=1), axis=1)  # 对输出按照单个Filter的值求和
        self.out = np.nan_to_num(np.exp(self.image) / temp[:, np.newaxis, np.newaxis])
        return self.out.reshape(-1, 1)  # softmax

    def backward(self, dout):
        self.dout = dout.reshape(np.shape(self.image))
        temp = np.sum(np.sum(self.dout, axis=1), axis=1)
        temp = temp[:, np.newaxis, np.newaxis]
        return (self.dout * temp - self.dout ** 2) / temp ** 2


class MaxPool:
    def __init__(self, image, pool_size, stride):
        self.out = None
        self.out_index = None
        self.image = image
        self.depth, self.height, self.width = image.shape
        self.pool_size = pool_size
        self.stride = stride

    def forward(self):
        self.out_height = (self.height - self.pool_size) // self.stride + 1
        self.out_width = (self.width - self.pool_size) // self.stride + 1
        out = np.zeros((self.depth, self.out_height, self.out_width))
        out_index = [[[0, ] * self.out_width] * self.out_height] * self.depth  # 用于记录最大值的位置
        for d in range(self.depth):
            for i in range(self.out_height):
                for j in range(self.out_width):
                    temp_matrix = self.image[d, i * self.stride:i * self.stride + self.pool_size,
                                  j * self.stride:j * self.stride + self.pool_size]  # 选取pool_size大小的矩阵，二维的
                    out[d, i, j] = np.max(temp_matrix)
                    out_index[d][i][j] = np.unravel_index(np.argmax(temp_matrix, axis=None), temp_matrix.shape)
        self.out = out
        self.out_index = out_index
        return out

    def backward(self, dout):
        dx = np.zeros_like(self.image)
        for d in range(self.depth):
            for i in range(self.out_height):
                for j in range(self.out_width):
                    index = self.out_index[d][i][j]
                    dx[d, i * self.stride + index[0], j * self.stride + index[1]] = dout[d, i, j]
        return dx


class LinearLayer:
    """
    从之前的BP算法中复制过来的
    """

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.threshold = np.random.randn(output_size, 1)
        # print(self.threshold)
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


def loss(y_true, y_pred):
    """
    交叉熵损失函数（Cross-Entropy Loss）
    :param y_true: 真实值
    :param y_pred: 模型输出值
    :return: 损失值大小
    """
    return -np.sum(y_true * np.log(y_pred))


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
tar_code = [1, 2, 1]
pic1 = get_tar_jpg(tar_code[0], tar_code[1], tar_code[2]).reshape(64, 64, 1)
test_con = ConvolutionLayer(3, 3, 1, 1, 1)
b = test_con.forward(pic1)
# Image.fromarray(b.reshape(64,64)).show()
test_maxpool = MaxPool(b, 2, 2)
c = test_maxpool.forward()
test_softmax = Softmax(c)
d = test_softmax.forward()
# np.count_nonzero(c == d)

# 查看data的value有几种类型
label_dict = data['value'].unique().reshape(-1, 1)
item_num = label_dict.shape[0]
full_connect = LinearLayer(d.shape[0], 15)

pre_out = full_connect.forward(d)

# 检查c的非零元素个数
# np.count_nonzero(c)
true_value = \
    data[(data["suite_id"] == tar_code[0]) & (data['sample_id'] == tar_code[1]) & (data['code'] == tar_code[2])][
        'value'].iloc[0]
tar_out = (label_dict[:, 0] == true_value).reshape(-1, 1)

total_loss = 0
# 计算损失
total_loss += loss(tar_out, pre_out)
final_out = label_dict[pre_out.argmax()][0]  # 模型最后的预测结果

fc_back = full_connect.backward(tar_out)
softmax_back = test_softmax.backward(fc_back)
maxpool_back = test_maxpool.backward(softmax_back)
convolution_back = test_con.backward(maxpool_back)
