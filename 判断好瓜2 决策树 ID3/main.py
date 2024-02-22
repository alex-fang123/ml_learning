import numpy as np
import pandas as pd
import my_functions as fc


data = pd.read_excel('raw_data.xlsx')
continuous_feature = {"密度", "含糖率"}

# 这是没有经过剪枝的树
my_tree = fc.GenerateTree(data, "好瓜", ["密度", "含糖率"])

# 储存决策树
np.save("my_tree.npy", my_tree)

# 读取决策树
# my_tree = np.load("my_tree.npy", allow_pickle=True).item()
# print(data.apply(fc.classify, axis=1, args=(my_tree,)))

# 用预剪枝的方法生成决策树
del data['密度']
del data['含糖率']
train = data.iloc[[0, 1, 2, 5, 6, 9, 13, 14, 15, 16], :]
test = data.iloc[[3, 4, 7, 8, 10, 11, 12],:]

# 这里训练出来的结果和西瓜书P81的结果不一样，
# 这是因为在第一次选择最优划分属性时，色泽和脐部属性在测试集上的正确率是一样的，我的模型用的色泽，书上用的脐部
my_tree1 = fc.GenerateTree(data, "好瓜", ["密度", "含糖率"], "pre", train, test)
