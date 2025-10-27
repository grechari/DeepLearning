import torch 
import torch.nn as nn 
import pandas as pd


# Решите задачу предсказания дохода по возрасту из датасет симпл

df = pd.read_csv('dataset_simple.csv')
X = torch.Tensor(df.iloc[:, 0].values).reshape(-1, 1)
y = torch.Tensor(df.iloc[:, 1].values).reshape(-1, 1)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, 1].values, marker='o') # грустно

class NNet_regression(nn.Module):
    
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, out_size) # просто сумматор
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# задаем параметры сети
inputSize = 1 # количество признаков задачи 
hiddenSizes = 32   #  число нейронов скрытого слоя 
outputSize = 1 # число нейронов выходного слоя

net = NNet_regression(inputSize,hiddenSizes,outputSize)

# В задачах регрессии чаще используется способ вычисления ошибки как разница квадратов
# как усредненная разница квадратов правильного и предсказанного значений (MSE)
# или усредненный модуль разницы значений (MAE)
lossFn = nn.L1Loss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

epoсhs = 1000

for i in range(epoсhs):
    pred = net(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%100==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net(X)

print('\nПредсказания:') # Иногда переобучается, нужно запускать обучение несколько раз
print(pred[0:10])
err = torch.mean(torch.abs(y - pred))# MAE - среднее отклонение от правильного ответа
print('\nОшибка (MAE): ')
print(err.item()) # измеряется в MPa


# Построим график
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, 1].values, marker='o')

with torch.no_grad():
    y1 = net.forward(torch.Tensor([20]))
    y2 = net.forward(torch.Tensor([60]))

plt.plot([20,60], [y1.numpy(),y2.numpy()],'r')
































































































# Пасхалка, кто найдет и сможет объяснить, тому +
# X = np.hstack([np.ones((X.shape[0], 1)), df.iloc[:, [0]].values])

# y = df.iloc[:, -1].values

# w = np.linalg.inv(X.T @ X) @ X.T @ y

# predicted = X @ w

# print(predicted)


























