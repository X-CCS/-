# coding=utf-8
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

#网络模型
class Net(nn.Module):
    def __init__(self):
        #定义Net的初始化函数，这个函数定义了该神经网络的基本结构
        super(Net, self).__init__() #复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.intohid_layer = nn.Linear(784, 100) #定义输入层到隐含层的连结关系函数
        self.hidtoout_layer = nn.Linear(100, 10)#定义隐含层到输出层的连结关系函数
    def forward(self, input):
        #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成
        x = torch.sigmoid(self.intohid_layer(input))   #输入input在输入层经过经过加权和与激活函数后到达隐含层
        x = torch.sigmoid(self.hidtoout_layer(x))       #类似上面
        return x

mnet = Net().cuda()
#数据集
train_dataset = dsets.MNIST(root = '../mnist/', #选择数据的根目录
                           train = True, # 选择训练集
                           transform = transforms.ToTensor(), # 转换成tensor变量
                           download = False) # 不从网络上download图片
test_dataset = dsets.MNIST(root = '../mnist/', # 选择数据的根目录
                           train = False, # 选择训练集
                           transform = transforms.ToTensor(),# 转换成tensor变量
                           download = False) # 不从网络上download图片
# 加载数据
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = 1,#每一次训练选用的数据个数
                                           shuffle = False)#将数据打乱
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = 1000,#每一次训练选用的数据个数
                                          shuffle = False)

loss_fn = torch.nn.MSELoss()#损失函数定义，可修改
optimizer = torch.optim.SGD(mnet.parameters(), lr = 0.1, momentum=0.9)

start = time.time()

for epoch in range(1):#训练次数
    print('current epoch = %d' % epoch)
    for i, (images, labels) in enumerate(train_loader): #利用enumerate取出一个可迭代对象的内容
        images = Variable(images.view(-1, 28 * 28).cuda())
        labels = Variable(labels.cuda())
        labels = torch.cuda.LongTensor(labels).view(-1,1)#将标签转为单列矩阵
        target= torch.zeros(1, 10).cuda().scatter_(dim = 1, index = labels, value = 0.98)#将标签转为onehot形式
        target+=0.01
        optimizer.zero_grad()               #清空节点值
        outputs = mnet(images)           #前向传播
        loss = loss_fn(outputs, target)  #损失计算
        loss.backward()                          #后向传播
        optimizer.step()                         #更新权值
        if i % 10000 == 0:
            print(i)
            total = 0
            correct = 0.0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28 * 28).cuda())
                outputs = mnet(images)                                        #前向传播
                _, predicts = torch.max(outputs.data, 1)                #返回预测结果
                total += labels.size(0)
                correct += (predicts == labels.cuda()).sum()
            print('Accuracy = %.2f' % (100 * float(correct) / total))

end = time.time()
print('花费时间%.2f' % (end - start))
