import torch.nn as nn
from node import  node

class convolution_block(nn.Module, node):
    all_convs = []
    node_type = 'conv'
    def __init__(self, in_h, in_w, in_channels, out_channels, kernel_size, padding = 0, stride = 1):
        try:
            assert(min(in_h, in_w) +2*padding >= kernel_size)
        except:
            raise inputSmallerThanKernel
        super(convolution_block, self).__init__()
        node.__init__(self, (in_channels, in_h, in_w))
        convolution_block.all_convs.append(self)
        self.in_channels  = in_channels
        self.out_channels = out_channels 
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.output_shape = self.out_shape()
        
        # NN Layers
        self.conv_layer = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.batch_norm = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

    def out_shape(self):
        c, h, w = self.input_shape
        C = self.out_channels
        H = (h + 2*self.padding - self.kernel_size)/self.stride + 1
        W = (w + 2*self.padding - self.kernel_size)/self.stride + 1
        return (C, H, W)
    
#     def determine_compatibility(self):
#         super(convolution_block, self).determine_compatibility()
#         self.compatible  = self.compatible and (len(self.in_adj) == 1)

    def describe_adj_list(self, in_adj, out_adj):
        super(convolution_block, self).describe_adj_list(in_adj, out_adj)
        try:
            assert(len(in_adj) == 1)
        except:
#             print(in_adj)
            raise Error('A convolution block can have only one in-edge')
    
    def remove(self): 
        ### Remove from all_convs list
        pass
    
    def __str__(self):
        return str(self.no) + " " + str(convolution_block.node_type) 
    
    def __repr__(self):
        return self.__str__()