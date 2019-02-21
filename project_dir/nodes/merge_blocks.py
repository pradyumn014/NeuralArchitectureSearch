import torch.nn as nn
from node import node

class merge_block(node):
    def __init__(self, parents, child):
        super(merge_block, self).__init__()
        self.describe_adj_list(parents, child)
        
class add_block(nn.Module, merge_block): 
    def __init__(self, parents, child):
        super(add_block, self).__init__()
        merge_block.__init__(self, parents, child)
        self.input_shape = self.in_adj[0].output_shape
        self.output_shape = self.out_shape()
                
    def forward(self, x, y):
        return x+y 
        
    def out_shape(self):
        return self.input_shape

class concat_block(merge_block):
    # Define Later
    pass

class convex_merge_block(merge_block):
    # Define Later
    pass