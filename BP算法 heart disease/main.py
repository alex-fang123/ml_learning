"""
    累计BP算法，用于心脏病数据集，数据集来自Kaggle，https://www.kaggle.com/datasets/mexwell/heart-disease-dataset
"""

import numpy as np
import pandas as pd


# %% 函数定义
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(x, weight1, weight2, threshold, threshold_output, learning_rate):
    # 第一层
    rep_x = np.tile(x, (hidden_layer_size, 1)).T
    hidden_layer_out = sigmoid(sum(rep_x[:-1, :] * weight1) - threshold.T)
    hidden_layer_out = hidden_layer_out.T

    # 第二层
    final_in = np.tile(hidden_layer_out, (output_layer_size, 1))
    output = sigmoid(sum(final_in * weight2) - threshold_output)

    g = output * (1 - output) * (x[-1] - output)  # 输出层神经元梯度
    e = hidden_layer_out * (1 - hidden_layer_out) * sum(g * weight2)  # 隐藏层神经元梯度
    weight2 += pd.DataFrame(learning_rate * g * hidden_layer_out)
    threshold_output += -learning_rate * g[0]
    weight1 += pd.DataFrame((learning_rate * np.tile(e, (1, 11)).T * rep_x[:-1, :]))
    threshold += -learning_rate * e
    return output - x[-1], weight1, weight2, threshold, threshold_output
    # return output - x[-1]

# %% 参数设置
# hidden_layer = 1
input_layer_size = 11
hidden_layer_size = 10
output_layer_size = 1
learning_rate = 0.1
threshold = np.random.rand(hidden_layer_size, 1)
threshold_output = np.random.rand(output_layer_size, 1)
weight1 = pd.DataFrame(np.random.rand(input_layer_size, hidden_layer_size))
weight2 = pd.DataFrame(np.random.rand(hidden_layer_size, output_layer_size))

# %% 读取数据
data = pd.read_csv('./archive/heart_statlog_cleveland_hungary_final.csv')

train = data.iloc[:int(len(data) * 0.8), :]
test = data.iloc[int(len(data) * 0.8):, :]

#%% 训练
a = test.apply(lambda x: forward(x, weight1, weight2, threshold, threshold_output, learning_rate), axis=1)
