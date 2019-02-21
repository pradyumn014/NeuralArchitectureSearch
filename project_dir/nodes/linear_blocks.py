import torch.nn as nn
from node import node

class FlattenBlock(nn.Module, node):
    def __init__(self, input_shape):
        super(FlattenBlock, self).__init__()
        node.__init__(self, input_shape, (input_shape[0]*input_shape[1]*input_shape[2],))
    def forward(self, x):
        return x.view(x.shape[0], -1)

class fullyConnectedBlock(nn.Module,node) :
    def __init__(self, input_units, output_units, activation = False) : 
        super(fullyConnectedBlock,self).__init__()
        node.__init__(self, (input_units, ), (output_units, ))
        self.layers = []
        self.layers.append(nn.Linear(input_units, output_units))
        if activation:
            self.layers.append(nn.ReLU())
     
    def forward(self, x) :
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
