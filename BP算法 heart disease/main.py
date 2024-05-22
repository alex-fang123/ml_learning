"""
    累计BP算法，用于心脏病数据集，数据集来自Kaggle，https://www.kaggle.com/datasets/mexwell/heart-disease-dataset
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LinearLayer:
    """
    定义好线性层就可以搭积木了
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


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.linear1 = LinearLayer(input_size, hidden_size)
        self.linear2 = LinearLayer(hidden_size, output_size)

    def forward(self, input):
        return self.linear2.forward(self.linear1.forward(input))

    def backward(self, dout):
        return self.linear1.backward(self.linear2.backward(dout))

    def update(self):
        self.linear1.update(self.linear1.learning_rate)
        self.linear2.update(self.linear2.learning_rate)


data = pd.read_csv('./archive/heart_statlog_cleveland_hungary_final.csv')
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['target']), data['target'], test_size=0.2)

x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

x_train = preprocessing.StandardScaler().fit_transform(x_train)
x_test = preprocessing.StandardScaler().fit_transform(x_test)

input_size = x_train.shape[1]
hidden_size = 20
output_size = 1
batch_size = 32

num_epochs = 100
mlp = MLP(input_size, hidden_size, output_size)
losses_train = []
losses_test = []

single_loss = 0
for epoch in range(num_epochs):
    if epoch % 10 == 0:
        print(f'Epoch {epoch}')
    for j in range(np.shape(y_train)[0] // batch_size + 1):
        if j != np.shape(y_train)[0] // batch_size:
            for i in range(j * batch_size, (j + 1) * batch_size):
                temp = x_train[i].reshape(-1, 1)
                mlp.forward(temp)
                single_loss += np.sum((y_train[i] - mlp.linear2.output) ** 2)
                mlp.backward(y_train[i] - mlp.linear2.output)
            mlp.update()
            losses_train.append(single_loss)
            single_loss = 0
        else:
            for i in range(j * batch_size, np.shape(y_train)[0]):
                temp = x_train[i].reshape(-1, 1)
                mlp.forward(temp)
                single_loss += np.sum((y_train[i] - mlp.linear2.output) ** 2)
                mlp.backward(y_train[i] - mlp.linear2.output)
            mlp.update()
            losses_train.append(single_loss)
            single_loss = 0

        # 在测试集上测试
        test_output = pd.DataFrame(x_test).apply(lambda x: mlp.forward(x.values.reshape(-1, 1))[0][0], axis=1)
        losses_test.append(np.sum((y_test - test_output) ** 2))

# 画图
dev_x = [i * 10 for i in range(len(losses_train))]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
line1 = ax1.plot(dev_x, losses_train, 'g-', label='train loss')
line2 = ax2.plot(dev_x, losses_test, 'r-', label='test loss')
ax1.set_xlabel('step count')
ax1.set_ylabel('train loss', color='g')
ax2.set_ylabel('test loss', color='r')
plt.show()
