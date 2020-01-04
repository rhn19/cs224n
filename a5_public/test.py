import torch

l = [[[0,1,2], [5,4,6]]]
t = torch.Tensor(l).permute(1,0,2)
print(t.size())
'''
d = dict()
d['<pad>'] = 0
d['{'] = 1
d['}'] = 2
print(d)
print(d['<pad>'])
'''

a = [1,2,3,4]
a = torch.cat(a)
print(a)
