# 使用预训练的ResNet对MNIST数据集进行分类

这段代码展示了如何使用预训练的ResNet模型对MNIST数据集进行分类。主要步骤包括数据准备、模型定义、训练与测试。

## 导入所需的库

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
```

## 定义超参数

这里我们设置了批处理的大小、选择设备（CPU或GPU）、设置训练的周期数。

```python
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
```

## 数据准备

我们使用`torchvision`提供的`datasets`和`transforms`模块来加载和预处理MNIST数据集。预处理操作包括调整图像大小以匹配ResNet的输入要求、转化为张量、标准化。

```python
pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
```

## 定义模型

我们使用了预训练的ResNet模型，但因为MNIST是灰度图像，ResNet的第一层需要从默认的3个输入通道改为1个。另外，MNIST有10个类别，所以我们修改了模型的最后一个全连接层以输出10个值。

```python
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(DEVICE)
```

## 定义优化器和损失函数

这里使用Adam优化器和交叉熵损失函数。

```python
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
```

## 定义训练函数

在每个周期中，我们让模型在每个批次的数据上进行训练，计算损失，然后进行反向传播和优化步骤。

```python
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## 定义测试函数

在每个周期结束后，我们在测试集上评估模型的性能。

```python
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0


    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
```

## 训练和测试模型

对于每个周期，我们先进行训练，然后进行测试。

```python
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, criterion, epoch)
    test(model, DEVICE, test_loader, criterion)
```

在每个周期结束后，我们都会打印测试集上的平均损失和准确度。



测试时效果不错

![image-20230610031619563](https://raw.githubusercontent.com/lijianye521/images/master/image-20230610031619563.png)

1. **损失**：训练损失和测试损失都非常低，这表明模型在学习过程中并没有遇到过拟合或者欠拟合的问题。在训练过程中，我们可以看到损失值是逐渐下降的，这是一个好的信号，说明模型在学习过程中逐步改善了其预测能力。
2. **准确度**：模型在测试集上的准确率稳定在99%，这是一个非常高的准确度，表明模型在分类任务上的性能非常出色。这个准确率意味着在所有的测试样本中，只有大约1%的样本被错误分类。
3. **训练过程**：从训练的输出中可以看出，模型在每个周期结束时，损失都有所下降。这表明模型在不断地学习和改善。有些周期里，损失值的下降可能不明显，这可能是因为模型已经接近最优解，进一步的改善已经很小。

总的来说，这是一个表现很好的模型，无论是从损失还是从准确度来看，都达到了很高的水平。这表明该模型已经很好地学习到了数据集的特征，能够准确地进行分类任务。