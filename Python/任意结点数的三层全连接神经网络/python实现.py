import math
import random
import numpy as np

IPNNUM=5     #输入层节点数
HDNNUM=10    #隐含层节点数
OPNNUM=3     #输出层节点数

class node:
    #结点类，用以构成网络
    def __init__(self,connectNum=0):
        self.value=0 #数值，存储结点最后的状态，对应到文章示例为X1，Y1等值
        self.W = (2*np.random.random_sample(connectNum)-1)*0.01

class net:
    #网络类，描述神经网络的结构并实现前向传播以及后向传播
    def __init__(self):
        #初始化函数，用于初始化各层间节点和偏置项权重
        #输入层结点
        self.inlayer=[node(HDNNUM)];
        for obj in range(1, IPNNUM):
            self.inlayer.append(node(HDNNUM))
        #隐含层结点
        self.hidlayer=[node(OPNNUM)];
        for obj in range(1, HDNNUM):
            self.hidlayer.append(node(OPNNUM))
        #输出层结点
        self.outlayer=[node(0)];
        for obj in range(1, OPNNUM):
            self.outlayer=[node(0)]

        self.yita = 0.1                                     #学习率η
        self.k1=random.random()                       #输入层偏置项权重
        self.k2=random.random()                       #隐含层偏置项权重
        self.Tg=np.zeros(OPNNUM)                   #训练目标
        self.O=np.zeros(OPNNUM)                     #网络实际输出

    def sigmoid(self,z):
        #激活函数
        return 1 / (1 + math.exp(-z))

    def getLoss(self):
        #损失函数
        loss=0
        for num in range(0, OPNNUM):
            loss+=pow(self.O[num] -self.Tg[num],2)
        return loss/OPNNUM

    def forwardPropagation(self,input):
        #前向传播
        for i in range(0, IPNNUM):
            #输入层节点赋值
            self.inlayer[i].value = input[i]
        for hNNum in range(0,HDNNUM):
             #算出隐含层结点的值
            z = 0
            for iNNum in range(0,IPNNUM):
                z+=self.inlayer[iNNum].value*self.inlayer[iNNum].W[hNNum]
            #加上偏置项
            z+= self.k1
            self.hidlayer[hNNum].value = self.sigmoid(z)
        for oNNum in range(0,OPNNUM):
            #算出输出层结点的值
            z = 0
            for hNNum in range(0,HDNNUM):
                z += self.hidlayer[hNNum].value* self.hidlayer[hNNum].W[oNNum]
            z += self.k2
            self.O[oNNum] = self.sigmoid(z)

    def backPropagation(self,T):
        #反向传播，这里为了公式好看一点多写了一些变量作为中间值
        for num in range(0, OPNNUM):
            self.Tg[num] = T[num]
        for iNNum in range(0,IPNNUM):
            #更新输入层权重
            for hNNum in range(0,HDNNUM):
                y = self.hidlayer[hNNum].value
                loss = 0
                for oNNum in range(0, OPNNUM):
                    loss+=(self.O[oNNum] - self.Tg[oNNum])*self.O[oNNum] * (1 - self.O[oNNum])*self.hidlayer[hNNum].W[oNNum]
                self.inlayer[iNNum].W[hNNum] -= self.yita*loss*y*(1- y)*self.inlayer[iNNum].value
        for hNNum in range(0,HDNNUM):
            #更新隐含层权重
            for oNNum in range(0,OPNNUM):
                self.hidlayer[hNNum].W[oNNum]-= self.yita*(self.O[oNNum] - self.Tg[oNNum])*self.O[oNNum]*\
                    (1- self.O[oNNum])*self.hidlayer[hNNum].value

    def printresual(self,trainingTimes):
        #信息打印
        loss = self.getLoss()
        print("训练次数：", trainingTimes)
        print("loss",loss)
        for oNNum in range(0,OPNNUM):
            print("输出",oNNum,":",self.O[oNNum])

#主程序
mnet=net()
input=np.array([0.1,0.2,0.3,0.4,0.5])
target=np.array([0.1,0.4,0.5])

for n in range(0,1000):
    mnet.forwardPropagation(input)
    mnet.backPropagation(target)
    if (n%200==0):
        mnet.printresual(n)
