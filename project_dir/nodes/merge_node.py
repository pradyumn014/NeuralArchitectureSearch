import torch.nn as nn
from node import  node

### Is is it required to be to a derived class of nn.Module ?
class merge_node(node):
    all_merge_nodes = []
    node_type = 'merge'    
    def __init__(self, parents, child):
        super(merge_node, self).__init__()
#         node.__init__(self)
        try:
            self.describe_adj_list(parents, child)
        except Exception as e:
            print e.expr
            raise e
        merge_node.all_merge_nodes.append(self)
        
    def describe_adj_list(self, in_adj, out_adj):
        super(merge_node, self).describe_adj_list(in_adj, out_adj)
        try:
            assert(len(in_adj) == 2)
        except:
            raise Error('Parents must be exactly two')
        
class add_node(nn.Module, merge_node):
    all_add_nodes = []
    node_type = 'add'    
    def __init__(self, parents, child):
        super(add_node, self).__init__()
        merge_node.__init__(self, parents, child)
        add_node.all_add_nodes.append(self)
        self.input_shape = self.in_adj[0].output_shape
        self.output_shape = self.out_shape()
                
    ### Does it allow to input paramteters ?
    def forward(self, x, y):
        return x+y 
        
    def out_shape(self):
        return self.input_shape

    def __str__(self):
        return str(self.no) + " " + str(add_node.node_type) 
    
    def __repr__(self):
        return self.__str__()
    
class concat_node(merge_node):
    # Define Later
    pass

class convex_merge_node(merge_node):
    # Define Later
    pass