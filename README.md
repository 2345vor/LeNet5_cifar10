LetNet-cifar10使用教程
# 1. 准备
## 1.1 下载cifar10数据集
> https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

解压放在date文件夹解压
新建cifar_img,放入你需要预测的图片，大小为3*32*32

## 1.2 安装相关依赖库
pip install 一下
> torch PIL tensorboardX torchvision
> imageio pickle 

# 2.调试网络
## 2.1 运行net.py

返回网络结构

```text
torch.Size([64, 10])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]           2,432
         MaxPool2d-2           [-1, 32, 16, 16]               0
            Conv2d-3           [-1, 32, 16, 16]          25,632
         MaxPool2d-4             [-1, 32, 8, 8]               0
            Conv2d-5             [-1, 64, 8, 8]          51,264
         MaxPool2d-6             [-1, 64, 4, 4]               0
           Flatten-7                 [-1, 1024]               0
            Linear-8                  [-1, 500]         512,500
            Linear-9                   [-1, 64]          32,064
           Linear-10                   [-1, 10]             650
================================================================
Total params: 624,542
Trainable params: 624,542
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.44
Params size (MB): 2.38
Estimated Total Size (MB): 2.84
----------------------------------------------------------------

Process finished with exit code 0
```
## 2.2 运行train.py
导入相关库---加载训练集和验证集---使用tensorboard记录日志---迭代训练---保存权重和日志
训练30轮，大约需要15分钟，精度达到76.01%，还行！第三十轮打印如下
```commandline
————————第30轮训练开始————————
545.7393746376038
训练次数:22700, Loss:0.5816277861595154
546.7848184108734
训练次数:22800, Loss:0.6448397040367126
547.836345911026
训练次数:22900, Loss:0.7739525437355042
548.8866136074066
训练次数:23000, Loss:0.48977798223495483
549.9396405220032
训练次数:23100, Loss:0.5046865940093994
550.9778225421906
训练次数:23200, Loss:0.7065502405166626
552.0327830314636
训练次数:23300, Loss:0.6883654594421387
553.0864734649658
训练次数:23400, Loss:0.54579758644104
整体测试集上的loss:541.3202195465565
整体测试集上的正确率:0.7601000070571899
save best model

Process finished with exit code 0

```
打开终端，在tensorboard中查看训练效果
>tensorboard --logdir "cifar_log"
> 训练

可视化训练损失、测试精度、测试损失
## 2.3 运行主程序main.py
导入相关库---加载需要预测的图片---加载训练权重---开始预测---显示预测结果
返回
```commandline
torch.Size([3, 32, 32])
class: plane        prob: 0.0737
class: car          prob: 0.808
class: bird         prob: 0.0039
class: cat          prob: 0.00127
class: deer         prob: 0.000887
class: dog          prob: 3.52e-05
class: frog         prob: 4.89e-06
class: horse        prob: 0.00747
class: ship         prob: 0.00797
class: truck        prob: 0.0965

Process finished with exit code 0

```

