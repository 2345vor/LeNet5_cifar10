# 搭建神经网络
import torch
from torch import nn
from torchsummary import summary
class cifar_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 500),
            nn.Linear(500, 64),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    test_cifar = cifar_model()
    input = torch.ones((64, 3, 32, 32))
    output = test_cifar(input)
    print(output.shape)  # torch.Size([64, 10]) # 检查模型的正确性
    device = torch.device('cuda:0')
    test_cifar.to(device)
    summary(test_cifar, (3, 32, 32))# 打印网络结构
