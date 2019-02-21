import torch.nn as nn
from node import node 

class convolution_block(nn.Module, node):
    def __init__(self, in_channels, in_h, in_w, out_channels, kernel_size, padding = 0, stride = 1):
        super(convolution_block, self).__init__()
        node.__init__(self, (in_channels, in_h, in_w))
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
    
    def refresh(self, in_channels, in_h, in_w, out_channels, kernel_size, padding = 0, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel_size  = kernel_size 
        self.stride = stride 
        self.padding = padding
        self.input_shape = (in_channels, in_h, in_w)
        self.output_shape = self.out_shape()
        
        ## NN Layers
        self.conv_layer = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.batch_norm = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
                
    def out_shape(self):
        c, h, w = self.input_shape
        C = self.out_channels
        H = (h + 2*self.padding - self.kernel_size)/self.stride + 1
        W = (w + 2*self.padding - self.kernel_size)/self.stride + 1
        return (C, H, W)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x 