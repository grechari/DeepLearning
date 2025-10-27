import torch 
import pandas as pd
import random
import torch.nn as nn

# Тензор x целочисленного типа, хранящий случайные значение
x = torch.randint(1, 10, (3,3))
print(x)

# Преобразовать тензор к типу float32
x = x.to(dtype=torch.float32)
x.requires_grad = True

# Возвести в степень n = 3
y=x**3
print(y)
# Умножить на случайное число от 1 до 10
z=y*random.uniform(1, 10)
print(z)
# Взять экспоненту от полученного числа
ex=torch.exp(z)
print(ex)
# Получить значение производной для полученного значения по x
ex.backward(torch.ones_like(ex))
print(x.grad) # градиенты d(ex)/dx


###########                                                        ############
###########    Обучение линейного алгоритма на основе нейронов    #############
###########                                                       #############


# На основе кода обучения линейного алгоритма создать код для решения задачи классификации цветков ириса

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
X = torch.tensor(df.iloc[:, 0:4].values, dtype=torch.float32) 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_labels = le.fit_transform(df.iloc[:, 4])
y = torch.zeros(150, 3, dtype=torch.float32)
y[torch.arange(150), y_labels] = 1 

linear = nn.Linear(4, 3)

print ('w: ', linear.weight)
print ('b: ', linear.bias)

lossFn = nn.MSELoss() 

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01) 

for i in range(0,20):
    optimizer.zero_grad() # забыли
    pred = linear(X)
    loss = lossFn(pred, y)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    loss.backward()
    optimizer.step()
    