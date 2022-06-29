from PIL import Image
import time

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from net import cifar_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imgae_path = "cifar_img/dog.jpg"
image = Image.open(imgae_path)
image = image.convert('RGB')
# 显示图片
plt.imshow(image)
transform = torchvision.transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
image = transform(image)
print(image.shape)  # 因为cifar网络要求输入32*32的图片
# 获取预测结果
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 加载模型并使用GPU
model = cifar_model().to(device)
# 加载 train.py 里训练好的模型
model.load_state_dict(torch.load("./cifar_weight/best_model.pth"))

image = torch.reshape(image, (1, 3, 32, 32))
# image = image.to(device)

model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(image.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

# 顶部显示预测类型和概率
print_res = "class: {}   prob: {:.3}".format(classes[predict_cla],
                                             predict[predict_cla].numpy())
plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(classes[i],
                                              predict[i].numpy()))
plt.show()


