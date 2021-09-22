import bayesfunc as bf

from torch import nn
from copy import deepcopy

def get_net(type, inducing_size):
    if (type == "global"):
        fc1 = bf.GILinear(in_features=784, out_features=100, inducing_batch=inducing_size, bias=True, full_prec=True)
        fc2 = bf.GILinear(in_features=100, out_features=100, inducing_batch=inducing_size, bias=True, full_prec=True)
        fc3 = bf.GILinear(in_features=100, out_features=10, inducing_batch=inducing_size, bias=True, full_prec=True)
        net = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)
        net = bf.InducingWrapper(net, inducing_batch=inducing_size, inducing_shape=(inducing_size, 784))
        return net
        
    elif (type == "factorised"):
        fc1 = bf.FactorisedLinear(in_features=784, out_features=100, bias=True)
        fc2 = bf.FactorisedLinear(in_features=100, out_features=100, bias=True)
        fc3 = bf.FactorisedLinear(in_features=100, out_features=10, bias=True)
        net = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)
        return net

def get_old_net(net):
    old_net = deepcopy(net)
    for param in old_net.parameters(): 
        param.requires_grad = False
    return old_net