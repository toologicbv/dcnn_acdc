import torch
import numpy as np
from torch.autograd import Variable
import socket
import torch
from tqdm import tqdm

my_var = Variable(torch.FloatTensor(2, 3))

print("Running on {}".format(socket.gethostname()))


def test_kwargs(**kwargs):
    for k, value in kwargs.iteritems():
        print("{}: {}".format(k, value))


def next_batch(iters=100):

    for i in np.arange(iters):
        yield i


# test_kwargs(prob1=100, prob2=1000)
for idx in next_batch(iters=5):
    print idx

