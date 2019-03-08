import numpy as np

A = np.array([1,2,3])
result = A + 100 # 不需要写A + [100, 100, 100]
print(result)  #结果 [101 102 103]


A = np.array([[1,2,3],[4,5,6]])
result = A + [100, 200, 300]
print(result)

