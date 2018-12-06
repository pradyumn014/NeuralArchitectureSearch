import torch.nn as nn
from node import  node

class max_pool_node(nn.Module, node):
    all_max_pools = []
    node_type = 'max_pool'    
    def __init__(self, in_h, in_w, in_channels, kernel_size, padding = 0, stride = 1):
        try:
            assert(min(in_h, in_w) +2*padding > kernel_size)
        except:
            raise inputSmallerThanKernel
        super(max_pool_node, self).__init__()    
        node.__init__(self, (in_channels, in_h, in_w))
        max_pool_node.all_max_pools.append(self)
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
    
#     def determine_compatibility(self):
#         super(max_pool_node, self).determine_compatibility()
#         self.compatible  = self.compatible and (len(self.in_adj) == 1)

    def describe_adj_list(self, in_adj, out_adj):
        super(max_pool_node, self).describe_adj_list(in_adj, out_adj)
        try:
            assert(len(in_adj) == 1)
        except:
            raise Error('A max-pool block can have only one in-edge')

    def remove(self):
        pass
    
    def __str__(self):
        return str(self.no) + " " + str(max_pool_node.node_type) 
    
    def __repr__(self):
        return self.__str__()