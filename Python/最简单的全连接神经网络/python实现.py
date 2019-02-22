import math

class node:
    # 结点类，用以构成网络
    def __init__(self, w1=None, w2=None):
        self.value = 0  # 数值，存储结点最后的状态，对应到文章示例为X1，Y1等值
        self.W = [w1, w2]  # 结点到下一层的权值

class net:
    # 网络类，描述神经网络的结构并实现前向传播以及后向传播
    # 这里为文章示例中的三层网络，每层结点均为两个
    def __init__(self):
        # 初始化函数，将权重，偏置等值全部初始化为博文示例的数值
        self.inlayer = [node(0.15, 0.25), node(0.20, 0.30)]  # 输入层结点
        self.hidlayer = [node(0.40, 0.50), node(0.45, 0.55)]  # 隐含层结点
        self.outlayer = [node(), node()]                  # 输出层结点

        self.yita = 0.5                                 # 学习率η
        self.k1 = 0.35                                  # 输入层偏置项权重
        self.k2 = 0.60                                  # 隐含层偏置项权重
        self.Tg = [0, 0]                                # 训练目标
        self.O = [0, 0]                                 # 网络实际输出

    def sigmoid(self, z):
        # 激活函数
        return 1 / (1 + math.exp(-z))

    def getLoss(self):
        # 损失函数
        return ((self.O[0] -self.Tg[0])**2+ (self.O[1] - self.Tg[1])**2)/2

    def forwardPropagation(self, input1, input2):
        # 前向传播
        self.inlayer[0].value = input1
        self.inlayer[1].value = input2
        for hNNum in range(0, 2):
             # 算出隐含层结点的值
            z = 0
            for iNNum in range(0, 2):
                z += self.inlayer[iNNum].value*self.inlayer[iNNum].W[hNNum]
            # 加上偏置项
            z += self.k1
            self.hidlayer[hNNum].value = self.sigmoid(z)

        for oNNum in range(0, 2):
            # 算出输出层结点的值
            z = 0
            for hNNum in range(0, 2):
                z += self.hidlayer[hNNum].value* self.hidlayer[hNNum].W[oNNum]
            z += self.k2
            self.outlayer[oNNum].value = self.sigmoid(z)
            self.O[oNNum] = self.sigmoid(z)


    def backPropagation(self,T1,T2):
        # 反向传播，这里为了公式好看一点多写了一些变量作为中间值
        # 计算过程用到的公式在博文中已经推导过了，如果代码没看明白请看看博文
        self.Tg[0] = T1
        self.Tg[1] = T2
        for iNNum in range(0, 2):
            # 更新输入层权重
            for wnum in range(0, 2):
                y = self.hidlayer[wnum].value
                self.inlayer[iNNum].W[wnum] -= self.yita*((self.O[0] - self.Tg[0])*self.O[0] *(1- self.O[0])*\
                    self.hidlayer[wnum].W[0] +(self.O[1] - self.Tg[1])*self.O[1] *(1 - self.O[1])*\
                    self.hidlayer[wnum].W[1])*y*(1- y)*self.inlayer[iNNum].value;

        for hNNum in range(0,2):
            #更新隐含层权重
            for wnum in range(0,2):
                self.hidlayer[hNNum].W[wnum]-= self.yita*(self.O[wnum] - self.Tg[wnum])*self.O[wnum]*\
                    (1- self.O[wnum])*self.hidlayer[hNNum].value;

    def printresual(self):
        #信息打印
        loss = self.getLoss()
        print("loss",loss)
        print("out1",self.O[0])
        print("out2",self.O[1])

#主程序
mnet=net();
for n in range(0,20000):
    mnet.forwardPropagation(0.05, 0.1)
    mnet.backPropagation(0.01, 0.99)
    if (n%1000==0):
        mnet.printresual()
