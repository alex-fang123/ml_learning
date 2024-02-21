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
print(fc.classify(my_tree, data.iloc[1, :]))
