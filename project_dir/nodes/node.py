class node(object):
    no =0
#     node_type = 'simple'
    def __init__(self, input_shape = (0, 0, 0), output_shape = (0, 0, 0)): # c, i1, i2
        node.no += 1
        self.no = node.no
        self.in_adj = []
        self.out_adj = []
        self.input_shape = input_shape
        self.output_shape = output_shape
    
#     def set_shape(self, input_shape, output_shape):
#         self.input_shape = input_shape
#         self.output_shape = output_shape
        

    def describe_adj_list(self, in_adj, out_adj):
        self.in_adj = in_adj
        self.out_adj = out_adj

    def out_shape(self):
        pass
    
    def remove(self):
        ## needed in graph class
        pass
    
    
    def __str__(self):
        return str(self.no) + " " + str(type(self)) 

    def __repr__(self):
        return self.__str__()