import torch.nn as nn
from node import node

class max_pool_block(nn.Module, node):  
    def __init__(self, in_channels, in_h, in_w, kernel_size, padding = 0, stride = 1):
        super(max_pool_node, self).__init__()    
        node.__init__(self, (in_channels, in_h, in_w))
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.output_shape = self.out_shape()
        
        ## NN Layer
        self.max_pool_layer = nn.MaxPool2d(self.kernel_size, self.stride, self.padding)
        
    def forward(self, x):
        return self.max_pool_layer(x)
    
    def out_shape(self):
        c, h, w = self.input_shape
        C = c
        H = (h + 2*self.padding - self.kernel_size)/self.stride + 1
        W = (w + 2*self.padding - self.kernel_size)/self.stride + 1
        return (C, H, W)