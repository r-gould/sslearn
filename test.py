"""import torch
bs = 2
encode_dim = 4
K = 6
q = torch.randn(bs, encode_dim)
k = torch.randn(bs, K+1, encode_dim)

print("QUERY:", q)
print("KEY:", k)

q = q.unsqueeze(1)
k = k.transpose(1, 2)

out = torch.bmm(q, k)
print(out.shape)
out = out.squeeze()
print(out.shape)
print(out)"""

"""import torchvision

print(torchvision.__version__)
import torch
print(torch.__version__)"""

"""import torch

avg = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
a = torch.zeros(4, 8, 2, 2)
b = torch.randint_like(a, 128)
print(b)
print(avg(b))"""

"""import torch
from archs import ResNet

a = torch.randn(4, 3, 32, 32)
model = ResNet(channels_in=3, encode_dim=128, model_name="resnet-34")
print(model(a).shape)"""

"""import torch
a = torch.randn(4, 3)
b = torch.nn.functional.normalize(a, dim=-1)
print(a)
print(b)"""

"""import torch

a = torch.randn(5, 4)
b = torch.randint(0, 5, (10,))

a_idxs = torch.argmax(a, dim=-1)
pred_labels = b[a_idxs]
labels = torch.randint(0, 5, (5,))
print(a_idxs)
print(b)
print(pred_labels)
print(torch.sum(pred_labels == labels).item())"""

"""import matplotlib.pyplot as plt

losses = [1,4,7,3,6,-8]
x = [i for i in range(1, len(losses)+1)]
plt.title("Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(x, losses)
plt.savefig("train.png")
plt.show()"""

"""class A:

    @property
    def metric_str(self):
        raise NotImplementedError()

a = A()
print(a.metric_str)"""

"""import torch
from torch.distributions import Categorical

a = Categorical(probs="hi")"""

"""import torch

loss_func = torch.nn.CrossEntropyLoss()
a = torch.randn(5, 10)
b = torch.randint(0, 10, (5,))
loss = loss_func(a, b)
print(isinstance(loss, torch.Tensor))"""

for a in range(5):
    a = 2 * a
    b = 3*a

print(a, b)
del a, b
print(a)
print(b)
