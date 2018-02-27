import torch
import numpy as np
from torch.autograd import Variable
import socket


my_var = Variable(torch.FloatTensor(2, 3))

print("Running on {}".format(socket.gethostname()))
