import random
import torch
from d2l import torch as d2l


# 1.获取数据集（这里是造一个)
def synthetic_data(W, b, num_examples):
    X = torch.normal(0., 1., (num_examples, len(W)))
    Y = torch.matmul(X, W) + b
    Y += torch.normal(0, 0.01, Y.shape)
    # print(Y)
    # print('reshape:')
    # print(Y.reshape(-1, 1))
    return X, Y.reshape((-1, 1))


# 2. 构建data_loader
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 3. 构建模型
def linreg(X, W, b):
    return torch.matmul(X, W) + b


# 4. 构建损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 5. 构建优化器
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    true_W = torch.tensor([2, -3.4])
    true_b = torch.tensor(5.7)
    features, labels = synthetic_data(true_W, true_b, 1000)
    batch_size = 10
    # for X, Y in data_iter(batch_size, features, labels):
    #     print(X)
    #     print(Y)
    #     exit()
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    # 6. 训练过程
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {train_l.sum()}')
    print('w:', w)
    print('b:', b)