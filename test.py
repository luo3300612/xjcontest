from main import Net
import torch

net = Net()

x = torch.randn((10,3,224,224))

out = net(x)

print(out.shape)
