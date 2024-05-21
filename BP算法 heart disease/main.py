"""
    累计BP算法，用于心脏病数据集，数据集来自Kaggle，https://www.kaggle.com/datasets/mexwell/heart-disease-dataset
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

class LinearLayer:
    """
    定义好线性层就可以搭积木了
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.threshold = np.random.randn(output_size, 1)
        self.input = None
        self.output = None
        self.delta = None
        self.delta_weights = None
        self.delta_bias = None

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
        self.delta = dout * self.output * (1 - self.output)
        self.delta_weights = np.dot(self.input, self.delta.T)
        dx = np.dot(self.weights, self.delta)
        self.weights += self.delta_weights
        self.threshold += -self.delta
        return dx

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.linear1 = LinearLayer(input_size, hidden_size)
        self.linear2 = LinearLayer(hidden_size, output_size)

    def forward(self, input):
        return self.linear2.forward(self.linear1.forward(input))

    def backward(self, dout):
        return self.linear1.backward(self.linear2.backward(dout))


data = pd.read_csv('./archive/heart_statlog_cleveland_hungary_final.csv')
label = data['target']
features = data.drop(columns=['target'])

features = preprocessing.StandardScaler().fit_transform(features)
input_size = features.shape[1]
hidden_size = 10
output_size = 1
batch_size = 1
