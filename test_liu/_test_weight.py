import torch
import pdb
import torch.nn.functional as F

net = torch.nn.Linear(1, 1)
print(net)
pdb.set_trace()
#print(net.named_parameters())
for param in net.named_parameters():
    print(param[0], param[1].data)
for param in net.named_parameters():
    print(param[0], param[1])
#weight tensor([[-0.4321]])
#bias tensor([0.6777])

#for param in net.parameters():
#    print(param.data)
#tensor([[-0.4321]])
#tensor([0.6777])
