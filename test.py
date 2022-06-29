from PIL import Image
import time

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


imgae_path = "./cifar_img/car.jpg"
image = Image.open(imgae_path)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
image = transform(image)
print(image.shape)  # 因为cifar网络要求输入32*32的图片


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
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

model = torch.load("./cifar_weight/best_model.pth")
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(device)

model.eval()
with torch.no_grad():
    output = model(image)
# print(output)

print(output.argmax(1))
