import h5py
import matplotlib.pyplot as plt
import numpy as np

def load_dataset():
    # 字典！以键值对的方式保存的一种数据结构
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5','r')
    test_dataset = h5py.File('./datasets/test_catvnoncat.h5','r')

    test_set_x = np.array(test_dataset["test_set_x"][:])
    train_set_x = np.array(train_dataset["train_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])



    plt.figure(figsize=(2,2))
    plt.imshow(train_set_x[110]) # 填充第10张图片
    plt.show()  # 显示


    # 64*64*3=12288
    #(209,64*64*3)->(12288,209)
    # train_set_x.shape[0] =29
    # -1代表64*64*3
    # T代表转置
    train_set_x = train_set_x.reshape(train_set_x.shape[0],-1).T  # (12288,209)
    test_set_x = test_set_x.reshape(test_set_x.shape[0],-1).T  # (12288,209)

    # （209，）->(1,209)
    #numpy中，把数组的维数成为"秩"，一维数组也称为秩为1的数组，二维数组也称为秩为2的数组，以此类推
    #转变训练集的维度
    train_set_y= train_set_y.reshape(1,train_set_y.shape[0])
    # 转变测试集的维度
    test_set_y= test_set_y.reshape(1,test_set_y.shape[0])

    return train_set_x,train_set_y,test_set_x,test_set_y

def init_parameters(fc_net):
    #1、定义一个字典，存放参数矩阵W1,b1,W2,b2,W3,b3,W4,b4,
    parameters={}
    layer_num=len(fc_net)  # layer_num=5 因为fc_net=[12288,4,3,2,1]

    # 第0层为输入层是没有参数的，所以从第一层开始遍历
    for L in range(1,layer_num):
        # np.random.randn 标准高斯分布中的随机选取一些数值
        parameters["W"+str(L)] = np.random.randn(fc_net[L],fc_net[L-1])*0.01
        parameters["b"+str(L)] = np.zeros((fc_net[L],1)) # zeros(shape, dtype=None, order='C')

    for L in range(1,layer_num):
        print("W"+str(L)+"=",parameters["W"+str(L)].shape)
        print("b"+str(L)+"=",parameters["b"+str(L)].shape)

    return parameters


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def forward_pass(A0,paramenters): #前向计算函数
    cache = {}  # 缓存
    A = A0
    # print(parameters)
    # print(len(parameters))
    layer_num = len(parameters)//2 # 写成/2代表精确除法，写成//2代表向下取整除法，得到一个整数
    for L in range(1,layer_num+1):
        # z = wx+b
        Z = np.dot(parameters["W"+str(L)],A) + parameters["b"+str(L)]
        # （4，12288）*（12288，209）+（4，1）=（4，209）+（4，1）
        A = sigmoid(Z)
        cache["A"+str(L)]= A
        cache["Z"+str(L)]= A
    return A,cache

if __name__=="__main__":

    #1.加载数据
    train_set_x, train_set_y, test_set_x, test_set_y=load_dataset()
    #2、对输入像素值做归一化（0~255）->(0~1)
    # 避免神经元输出现象饱和
    train_set_x = train_set_x/255.
    test_set_x = train_set_x/255.
    # 3、定义全连接神经网络各层的神经个数，并初始化参数w和b
    # 输入第0层是122888，自定义第一层，第二层，第三层，最后一层是二分类所以为1
    fc_net=[12288,4,3,2,1]
    parameters = init_parameters(fc_net)

    # 前向过程求和z = wx+b ;激活a= f(z)
    AL,cache = forward_pass(train_set_x,parameters) # AL=(1,209)
    print("AL.shape =",AL.shape)
    print("AL=",AL) # 矩阵元素值应该介于0~1,因为sigmoid函数输出的是概率值