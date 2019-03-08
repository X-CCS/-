import torch
import torch.nn as nn
from torch.autograd import Variable


m = nn.Linear(20, 30)
input = Variable(torch.randn(128, 20))
output = m(input)
print(output.size())