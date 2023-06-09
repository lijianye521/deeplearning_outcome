# pytorch图像识别之手写字识别

## 初始化一个张量

```python
torch.tensor([[1., -1.], [1., -1.]])
torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
```

硬性翻译：参考网站https://pytorch.org/docs/stable/tensors.html#initializing-and-basic-operations

## 专业名词解释

#### **1.参数与超参数**

参数：模型f(x,θ）中θ为模型参数，可以通过优化算法进行学习

超参数：用来定义模型结构或优化策略。

#### **2.batch_size**

每次处理的数据数量。

#### **3.epoch轮次**

把一个数据集，循环运行多少轮。

#### **4.transforms变换**

主要讲图片转化为tensor，旋转图片，以及正则化。

#### **5.nomalize正则化**

模型出现过拟合现象时，降低模型复杂度。

#### **6.卷积层**

由卷积核构建，卷积核简称为卷积，也称为滤波器，卷积的大小可以在实际需要时自定义其长和宽

#### **7.池化层**

对图片进行压缩(降采样)的一种方法，如max pooling average pooling 等。

#### **8.激活层**

激活函数的作用就是，在所有的隐藏层之间添加一个激活函数，这样的输出就是一个非线性函数了，因而神经网络的表达能力更加强大

#### **9.损失函数**

在深度学习中，损失反应模型最后预测结果与实际真值之间的差距，可以用来分析训练过程的好坏，模型是否收敛等，例如均等损失，交叉熵损失等。

## 手写字体的识别流程

1. 定义超参数
2. 构建transforms，主要对图像做变换
3. 下载、加载数据集MNIST
4. 构建网络模型法
5. 定义训练方法
6. 定义测试方法
7. 开始训练模型，输出预测结果

### 



## 创建环境

```shell
conda create -n pytorch_env python=3.8
conda activate pytorch_env
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```



### 代码

```python
#加载必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

#定义超参数
BATCH_SIZE=16#每批处理的数据
DEVICE=torch.device("cuda"if torch.cuda.is_available()else"cpu")#用cpu还是gpu
EPOCHS=10#训练数据集的轮次

#构建pipline 对图像做处理
pipeline =transforms.Compose([
    transforms.ToTensor(),#将图片转换成tensor
    transforms.Normalize((0.1307,),(0.3081,))#正则化 降低模型复杂度
  ])

#下载、加载数据集
from torch.utils.data import DataLoader

#下载数据集
train_set=datasets.MNIST("data",train=True,download=True,transform=pipeline)
test_set=datasets.MNIST("data",train=False,download=True,transform=pipeline)
#加载数据
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)#shuffle是打乱的意思
test_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)

#构建网络模型
class Digit(nn.Module):#继承Moudle类
    def __init__(self):
        super().__init__()
        self.convl=nn.Conv2d(1,10,5)#二维卷积  1：灰度图片的通道 10：输出通道 5：卷积核kernl
        self.conv2=nn.Conv2d(10,20,3)#10:输入通道20 ：输出通道 3：卷积核大小
        #全连接层  线性层
        self.fcl=nn.Linear(20*10*10,500)##20*10*10输入通道 500输出通道
        self.fc2=nn.Linear(500,10)#500输入通道  10输出通道
    def forward(self,x):
        input_size=x.size(0)#batch_size
        x=self.convl(x)  #输入：batch*1*28*28，输出：batch*10*24*24（28-5+1）
        x=F.relu(x)#保持shape不变 激活层
        x=F.max_pool2d(x,2,2)#池化层  对图片进行压缩   输入：batch*10*24*24
        #输出 batch*10*12*12
        x=self.conv2(x)#输入：batch*10*12*12  输出：batch*20*（12-3+1）-（12-3+1）
        x=F.relu(x)
        #拉伸
        x=x.view(input_size,-1)  #-1  自动计算维度20*10*10=2000
        #进入全连接层
        x=self.fcl(x)#输入：batch*2000  输出batch*500
        x=F.relu(x)
        x=self.fc2(x)#输入batch*500  输出：batch*10
        output=F.log_softmax(x,dim=1)#计算分类，每个数字的概率值
        return output

#定义优化器
model=Digit().to(DEVICE)
optimizer = optim.Adam(model.parameters())
#定义训练方法
def train_model(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_index,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)#部署到device上去
         #梯度初始化为0
        optimizer.zero_grad()
        #训练后的结果
        output=model(data)
        #计算损失
        loss=F.cross_entropy(output,target)#交叉熵损失函数
        #找到概率值最大的下标
        pred=output.max(1,keepdim=True)
        #反向传播
        loss.backward()
        #参数优化
        optimizer.step()
        if batch_index%3000==0:
            print("Train Epoch : {} \t Loss:{:.6f}".format(epoch,loss.item()))

#定义测试方法
def test_model(model,device,test_loader):
    #模拟验证
    model.eval()
    #正确率
    correct=0.0
    #测试损失
    test_loss=0.0
    with torch.no_grad():#不会计算梯度，也不会进行反向传播
        for data, target in test_loader:
            #部署到device上
            data,target=data.to(device),target.to(device)
            #测试数据
            output=model(data)
            #测试损失
            test_loss+=F.cross_entropy(output,target).item()
            #找到概率值最大的索引
            pred=output.max(1,keepdim=True)[1]#值 索引
            #另外两种方法 pred=torch.max(output,dim=1)
            #pred=output.argmax(dim=1)
            #累计正确的数目
            correct+=pred.eq(target.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print("Test-Average loss:{:.4f},Accuracy:{:.3f}\n".format(test_loss,100.0*correct/len(test_loader.dataset)))

#调用方法
for epoch in range(1,EPOCHS+1):
    train_model(model,DEVICE,train_loader,optimizer,epoch)
    test_model(model,DEVICE,test_loader)
```







## 运行结果

> C:\ana\envs\pytorch\python.exe D:/pythonProject1/main.py
> Train Epoch : 1 	 Loss:2.325215
> Train Epoch : 1 	 Loss:0.001763
> Test-Average loss:0.0023,Accuracy:98.838
>
> Train Epoch : 2 	 Loss:0.003396
> Train Epoch : 2 	 Loss:0.000558
> Test-Average loss:0.0015,Accuracy:99.257
>
> Train Epoch : 3 	 Loss:0.038988
> Train Epoch : 3 	 Loss:0.009089
> Test-Average loss:0.0010,Accuracy:99.525
>
> Train Epoch : 4 	 Loss:0.013535
> Train Epoch : 4 	 Loss:0.000193
> Test-Average loss:0.0006,Accuracy:99.665
>
> Train Epoch : 5 	 Loss:0.000430
> Train Epoch : 5 	 Loss:0.065546
> Test-Average loss:0.0009,Accuracy:99.478
>
> Train Epoch : 6 	 Loss:0.000847
> Train Epoch : 6 	 Loss:0.000007
> Test-Average loss:0.0004,Accuracy:99.787
>
> Train Epoch : 7 	 Loss:0.001314
> Train Epoch : 7 	 Loss:0.000021
> Test-Average loss:0.0004,Accuracy:99.785
>
> Train Epoch : 8 	 Loss:0.000016
> Train Epoch : 8 	 Loss:0.000010
> Test-Average loss:0.0003,Accuracy:99.853
>
> Train Epoch : 9 	 Loss:0.000063
> Train Epoch : 9 	 Loss:0.000112
> Test-Average loss:0.0004,Accuracy:99.847
>
> Train Epoch : 10 	 Loss:0.000394
> Train Epoch : 10 	 Loss:0.003856
> Test-Average loss:0.0004,Accuracy:99.817