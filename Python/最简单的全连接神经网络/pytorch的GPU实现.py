import time
import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        #定义Net的初始化函数，这个函数定义了该神经网络的基本结构
        super(Net, self).__init__() #复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.intohid_layer = nn.Linear(2, 2); #定义输入层到隐含层的连结关系函数
        self.hidtoout_layer = nn.Linear(2, 2);#定义隐含层到输出层的连结关系函数

    def forward(self, input):
        #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成
        x = torch.nn.functional.sigmoid(self.intohid_layer(input))    #输入input在输入层经过经过加权和与激活函数后到达隐含层
        x = torch.nn.functional.sigmoid(self.hidtoout_layer(x))       #类似上面
        return x

mnet = Net().cuda()  # 使用GPU服务器
target=Variable(torch.cuda.FloatTensor([0.01, 0.99]));   #目标输出
input=Variable(torch.cuda.FloatTensor([0.05, 0.01]));    #输入

loss_fn = torch.nn.MSELoss();                            #损失函数定义，可修改
optimizer = torch.optim.SGD(mnet.parameters(), lr=0.5, momentum=0.9);

start = time.time()

for t in range(0,5000):
    optimizer.zero_grad();      #清空节点值
    out=mnet(input);            #前向传播
    loss = loss_fn(out,target); #损失计算
    loss.backward();            #后向传播
    optimizer.step();           #更新权值

print(out.cpu());
end = time.time()
print(end - start)
