#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, embed_size: int):
        super(Highway, self).__init__()
        self.embed_size = embed_size

        self.x_proj = nn.Linear(self.embed_size, self.embed_size)
        #print(self.x_proj.weight, self.x_proj.bias)
        self.x_gate = nn.Linear(self.embed_size, self.embed_size)
        #print(self.x_gate.weight, self.x_gate.bias)

    def forward(self, x_conv: torch.Tensor)->torch.Tensor:
        proj = F.relu(self.x_proj(x_conv))
        #print(proj.size())
        #print(proj)
        gate = torch.sigmoid(self.x_gate(x_conv))
        #print(gate.size())
        #print(gate)
        highway = torch.mul(gate, proj) + torch.mul((1 - gate), x_conv)
        #print(highway.size())
        #print(highway)
        return highway

### END YOUR CODE

EMBED_SIZE = 4
h = Highway(EMBED_SIZE)
x = torch.Tensor([1,2,3,4])
print(x.size())
h.forward(x)
