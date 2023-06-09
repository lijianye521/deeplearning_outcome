# 深度学习团队成果

# 网络模型：Digit

Digit是一个由卷积神经网络（CNN）构成的模型，用于进行手写数字识别。人话就是自己写的一个网络。

## 网络架构

该模型首先通过两个卷积层处理输入的图像，然后通过两个全连接层进行最终的分类。模型的具体构造如下：

- 两个卷积层，其中第一个卷积层的输入是1个通道（因为是灰度图像），输出是10个通道，卷积核大小为5。第二个卷积层的输入是10个通道，输出是20个通道，卷积核大小为3。

- 然后是两个全连接层。第一个全连接层的输入是20*10*10，输出是500；第二个全连接层的输入是500，输出是10（对应10个数字类别）。

- 在卷积层和全连接层之间，使用了Relu激活函数和Max Pooling层。这种方式可以增加模型的非线性，同时减小参数的数量，降低过拟合的风险。

## 训练和测试方法

我们定义了训练和测试的方法，分别为`train_model`和`test_model`。在训练中，我们使用了Adam优化器和交叉熵损失函数。在每个epoch中，我们都会打印出训练损失。测试方法中，我们计算了测试数据上的平均损失和正确率。

下面是网络的参数

> Model Summary: ----------------------------------------------------------------        Layer (type)               Output Shape         Param # ================================================================            Conv2d-1           [-1, 10, 24, 24]             260            Conv2d-2           [-1, 20, 10, 10]           1,820            Linear-3                  [-1, 500]       1,000,500
>
> ...
>
> Forward/backward pass size (MB): 0.06 Params size (MB): 3.84 Estimated Total Size (MB): 3.91

## 训练结果评价

从训练结果看，模型的表现非常好。在10个训练周期之后，测试集上的平均损失降低到了0.0003，准确率达到了99.848%。这表明我们的模型学习到了手写数字的有效特征，并能很好地进行分类。

然而，值得注意的是，由于我们使用的是MNIST数据集，该数据集相对简单，数字清晰，背景单一。因此，这个网络在复杂环境或更难的手写数字识别任务上可能会有不同的表现。

在使用该模型的过程中，可能还需要考虑一些其他因素，如过拟合问题，特别是在数据集较小或类别不均衡的情况下。在这种情况下，可以尝试添加一些正则化手段如Dropout，或者使用数据增强等方法来提高模型的泛化能力。

> Train Epoch : 1 	 Loss:2.313934 Train Epoch : 1 	 Loss:0.012564 Test-Average loss:0.0025,Accuracy:98.748 Train Epoch : 2 	 Loss:0.040462 Train Epoch : 2 	 Loss:0.024586 Test-Average loss:0.0022,Accuracy:98.953 Train Epoch : 3 	 Loss:0.008967 Train Epoch : 3 	 Loss:0.014208 Test-Average loss:0.0030,Accuracy:98.552 Train Epoch : 4 	 Loss:0.003008 Train Epoch : 4 	 Loss:0.000006 Test-Average loss:0.0013,Accuracy:99.337 Train Epoch : 5 	 Loss:0.001304 Train Epoch : 5 	 Loss:0.000307 Test-Average loss:0.0005,Accuracy:99.742 Train Epoch : 6 	 Loss:0.000079 Train Epoch : 6 	 Loss:0.000009 Test-Average loss:0.0007,Accuracy:99.670 Train Epoch : 7 	 Loss:0.078984 Train Epoch : 7 	 Loss:0.000000 Test-Average loss:0.0008,Accuracy:99.585 Train Epoch : 8 	 Loss:0.000066 Train Epoch : 8 	 Loss:0.000000 Test-Average loss:0.0003,Accuracy:99.853 Train Epoch : 9 	 Loss:0.000000 Train Epoch : 9 	 Loss:0.440672 Test-Average loss:0.0004,Accuracy:99.827 Train Epoch : 10 	 Loss:0.000000 Train Epoch : 10 	 Loss:0.000003 Test-Average loss:0.0003,Accuracy:99.848
