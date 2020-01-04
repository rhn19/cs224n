
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, filters, char_embed_size, max_word_length, kernel_size = 5):  
        super(CNN, self).__init__()  
        self.kernel_size = kernel_size 
        self.max_word_length = max_word_length   
        self.conv_layer = nn.Conv1d(
            in_channels = char_embed_size,
            out_channels = filters,
            kernel_size = self.kernel_size
        )

    def forward(self, x_res: torch.Tensor)->torch.Tensor:
        x_conv = self.conv_layer(x_res)
        x_conv = F.max_pool1d(F.relu_(x_conv), self.max_word_length - self.kernel_size + 1).squeeze()
        return x_conv

### END YOUR CODE
