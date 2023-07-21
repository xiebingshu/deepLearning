import torch
x = torch.arange(12)
X = x.reshape(3, 4)
print(torch.zeros(2, 3, 4))
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)