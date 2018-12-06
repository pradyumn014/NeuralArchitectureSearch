class node(object):
    nodes = []
    node_type = 'simple'
    def __init__(self, input_shape = (0, 0, 0), output_shape = (0, 0, 0)): # c, i1, i2
        node.nodes.append(self)
        self.no  = len(node.nodes)
        self.in_adj = []
        self.out_adj = []
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.compatible = True
            
    def node_alright(self, curr_node):
        try:
            assert(issubclass(type(curr_node), node))
        except:
            raise Error('Not a node' + str(type(curr_node)))
# Put this section in graph class
#         try:
#             assert(curr_node in graph_nodes)
#         except:
#             raise nodeDoesNotExist
    
    def determine_compatibility(self):
        for curr_node in self.in_adj:
            curr = (curr_node.output_shape == self.input_shape)
            self.compatible  = self.compatible and curr
            
        for curr_node in self.out_adj:
            curr = (curr_node.input_shape == self.output_shape)
            self.compatible = self.compatible and curr

    def describe_adj_list(self, in_adj, out_adj):
        assert isinstance(in_adj, list), 'in_adj must be a list'
        assert isinstance(in_adj, list), 'out_adj must be a list'
        for curr_node in in_adj + out_adj:
            try:
                self.node_alright(curr_node)
            except Exception as e:
                raise e
        self.in_adj = in_adj
        self.out_adj = out_adj

    def out_shape(self):
        pass
    
    def remove(self):
        ## needed in graph class
        pass
    
    
    def __str__(self):
        return str(self.no) + " " + str(node.node_type) 

    def __repr__(self):
        return self.__str__()