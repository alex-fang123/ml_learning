import torch
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv('./archive/heart_statlog_cleveland_hungary_final.csv')
label = data['target']
features = data.drop(columns=['target'])

features = preprocessing.StandardScaler().fit_transform(features)
input_size = features.shape[1]
hidden_size = 10
output_size = 1
batch_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size).to(device),
    torch.nn.Sigmoid().to(device),
    torch.nn.Linear(hidden_size, output_size).to(device),
    torch.nn.Sigmoid().to(device)
).to(device)
cost = torch.nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.SGD(my_nn.parameters(), lr=0.1)

losses = []
for i in range(10):
    for j in range(len(features)):
        x = torch.tensor(features[j], dtype=torch.float32).to(device)
        y = torch.tensor(label[j], dtype=torch.float32).to(device)
        y_pred = my_nn(x)
        loss = cost(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if j % 10000 == 0:
            print(f'Epoch {i}, loss: {loss.item()}')
dev_x = [i * 10 for i in range(11900)]
plt.xlabel('step count')
plt.ylabel('loss')
plt.xlim((0, 200))
plt.ylim((0, 1000))
plt.plot(dev_x, losses)
plt.show()
