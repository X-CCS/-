import h5py
import matplotlib.pyplot as plt
import numpy as np

# 字典！以键值对的方式保存的一种数据结构
train_dataset = h5py.File('./datasets/train_catvnoncat.h5','r')
test_dataset = h5py.File('./datasets/test_catvnoncat.h5','r')

# print('train_dataset.keys()：',train_dataset.keys())
# print('test_dataset.keys()：',test_dataset.keys())
#
# for key in train_dataset.keys():
#     print('key：',key)
#     print('train_dataset[key]:',train_dataset[key])
#
# for key in test_dataset.keys():
#     print('key：',key)
#     print('test_dataset[key]:',test_dataset[key])


# train_set_x = np.array(train_dataset["train_set_x"][:])
# print(train_set_x)
# train_set_y = np.array(train_dataset["train_set_y"][:])
# print(train_set_y)

test_set_x = np.array(test_dataset["test_set_x"][:])
train_set_x = np.array(train_dataset["train_set_x"][:])
# print(test_set_x)
test_set_y = np.array(test_dataset["test_set_y"][:])
train_set_y = np.array(train_dataset["train_set_y"][:])
# print(test_set_y)

# print('train_set_x.shape = ',train_set_x.shape)
# print('test_set_x.shape = ',test_set_x.shape)
# print('train_set_y.shape = ',train_set_y.shape)
# print('test_set_y.shape = ',test_set_y.shape)

# plt.figure(figsize=(2,2))
# plt.imshow(train_set_x[10]) # 填充第10张图片
# plt.show()  # 显示

# plt.figure(figsize=(2,2))
# plt.imshow(train_set_x[110]) # 填充第10张图片
# plt.show()  # 显示
#
# plt.figure(figsize=(2,2))
# plt.imshow(train_set_x[200]) # 填充第10张图片
# plt.show()  # 显示

# 64*64*3=12288
#(209,64*64*3)->(12288,209)
# train_set_x.shape[0] =29
# -1代表64*64*3
# T代表转置
train_set_x = train_set_x.reshape(train_set_x.shape[0],-1).T  # (12288,209)
print('train_set_x.shape',train_set_x.shape)
test_set_x = test_set_x.reshape(test_set_x.shape[0],-1).T  # (12288,209)
print('test_set_x.shape',test_set_x.shape)

# train_set_y.shape =  (209,) 一维数组，明显不是，输出得是（1，209）才可以
# print('train_set_y.shape = ',train_set_y.shape)


# （209，）->(1,209)
#numpy中，把数组的维数成为"秩"，一维数组也称为秩为1的数组，二维数组也称为秩为2的数组，以此类推
#转变训练集的维度
train_set_y= train_set_y.reshape(1,train_set_y.shape[0])
print('train_set_y.shape =',train_set_y.shape) # train_set_y.shape = (1, 209)
# 转变测试集的维度
test_set_y= test_set_y.reshape(1,test_set_y.shape[0])
print('test_set_y.shape =',test_set_y.shape) # test_set_y.shape = (1, 50)