import pandas as pd
from statsmodels.api import add_constant
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
# https://www.zhihu.com/question/588306900
# 上面这个链接解释了两个包得到的参数结果为什么不一样，主要是参数求解方法和是否正则化的差别

#%% 用statsmodels
data = pd.read_excel('raw_data.xlsx')
y = data.好瓜
x = data.iloc[:, 1:3]
x = add_constant(x)

# 调包
model = Logit(y, x).fit()
print(model.summary())
coef = model.params

#%% sklearn 感觉sklearn的文档写得更好一些
model1 = LogisticRegression(penalty=None)
# x = data.iloc[:, 1:3]  # sklearn默认自带常数项
model1.fit(x, y)
coef1 = model1.coef_
print(coef1)

model2 = LogisticRegression(penalty="l2")
# x = data.iloc[:, 1:3]  # sklearn默认自带常数项
model2.fit(x, y)
coef2 = model2.coef_
print(coef2)

model3 = LogisticRegression(penalty=None, solver="newton-cg")
# x = data.iloc[:, 1:3]  # sklearn默认自带常数项
model3.fit(x, y)
coef3 = model2.coef_
print(coef3)

print(model1.predict(x), model2.predict(x), model3.predict(x))
