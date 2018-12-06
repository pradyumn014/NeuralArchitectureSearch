import torch.nn as nn

from nodes.node import node
from nodes.convolution_block import convolution_block
from nodes.max_pool_node import max_pool_node
from nodes.merge_node import merge_node


class Network(nn.Module, object):
    def __init__(self):
        super(Network, self).__init__()
        self.adj_mat = {}
        self.adj_list = {}
        self.nodes = []
        self.int_to_node = {}
        self.node_to_int = {}
        self.conv_blocks = []
        self.max_pool_blocks = [] # Change naming conv maybe ?
        self.topsort = []
        self.rank_in_topsort = {}
        self.max_no = 0
        self.node_dict = {}
        
        #self.fc = nn.Linear(3 * 28 * 28, 10)
        
    def __init__(self, adj_list, int_to_node):
        super(Network, self).__init__()
        assert isinstance(int_to_node, dict), 'int_to_node must be a dictionary'
        for _, cnode in int_to_node.items():
            assert isinstance(cnode, node), 'mapping in int_to_node should be to a node'
        
        assert isinstance(adj_list, dict), 'adj_list should be a dictionary'
        assert(len(int_to_node) == len(adj_list))
        for cnode, li in adj_list.items():
            assert cnode in int_to_node, 'mismatch between int_to_node and adj_list'
            try:
                assert(isinstance(li, list))
                assert(len(li) == 2)
                assert(isinstance(li[0], list) and isinstance(li[1], list))
            except:
                raise Error('Each mapping in adj_list should be to a two-dim list')
            for child_node in li[0]:
                assert child_node in int_to_node, 'mismatch between int_to_node and adj_list'
            for child_node in li[1]:
                assert child_node in int_to_node, 'mismatch between int_to_node and adj_list'

        self.adj_list = adj_list
        self.adj_mat = self.get_adj_mat(self.adj_list)
        self.nodes = int_to_node.keys()
        self.int_to_node = int_to_node
        self.node_dict = {}
        self.node_to_int = self.get_node_to_int(self.int_to_node)
        self.max_no = max(self.int_to_node)
        self.conv_blocks, self.max_pool_blocks = self.get_conv_and_max_pool_blocks()
        self.topsort = []
        self.rank_in_topsort = {}
        self.topsorting()
        # hardcoded part
        #self.fc = torch.nn.Linear(3 * 28 * 28, 10)
        
        
    def createModel(self):
         self.node_dict = torch.nn.ModuleDict(self.node_dict)
         #print(self.node_dict)
        
    def forward(self, x):
        self.topsorting() # though this is not required here
        outputs = {}
#         outputs[self.topsort[0]] = self.int_to_node[self.topsort[0]].forward(x)
#       This is a bit inconsistent with the design patt
        
       # print 'self.topsort ',self.topsort
        
        outputs[self.topsort[0]] = x # instead add identity forward function to node class for dummy object
        for ind in range(1, len(self.topsort)):
            node_no = self.topsort[ind]
            curr_node = self.int_to_node[node_no]
            outputs[node_no] = curr_node.forward(*map(lambda x: outputs[x], self.adj_list[node_no][0]))
        return outputs[self.topsort[-1]]
    
    def get_node_to_int(self, int_to_node):
        node_to_int = {}
        for no, cnode in int_to_node.items():
            node_to_int[cnode] = no
        return node_to_int

        
    def get_adj_mat(self, adj_list):
        adj_mat = {}
        nodes = adj_list.keys()
        for x in nodes:
            adj_mat[x] = {}
            for y in nodes:
                adj_mat[x][y] = 0
        for cnode, li in adj_list.items():
            for par in li[0]:
                adj_mat[par][cnode] = 1
            for child in li[1]:
                adj_mat[cnode][child] = 1
        return adj_mat
    
    def get_conv_and_max_pool_blocks(self):
        conv_blocks = []
        max_pool_blocks = []
        for x in self.nodes:
            if isinstance(self.int_to_node[x], convolution_block):
                conv_blocks.append(x)
                self.node_dict[str(x)]=self.int_to_node[x]
            elif isinstance(self.int_to_node[x], max_pool_node):
                max_pool_blocks.append(x)
                self.node_dict[str(x)]=self.int_to_node[x]
        return (conv_blocks, max_pool_blocks)
    
    def topsorting(self):
        # level problem
        topsort = []
        import Queue
        in_deg = {}
        q = Queue.Queue()
        for node in self.nodes:
            val  = len(self.adj_list[node][0])
#             val = len(self.int_to_node[node].in_adj)
            if val == 0:
                q.put(node)
            in_deg[node] = val
            
        while not q.empty():
            curr_node = q.get()
            topsort.append(curr_node)
            for child in self.adj_list[curr_node][1]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    q.put(child)
        self.topsort = topsort
        self.set_rank_in_topsort()
    
    def set_rank_in_topsort(self):
        for ind, node_no in enumerate(self.topsort):
            self.rank_in_topsort[node_no]  = ind
    
    def add_nodes_to_network(self, nodes):
        ### loophole here, assumption is that all changed nodes are being provided to the function
        ### for now, lets go on with it, but its an issue
        for curr_node in nodes:
            curr_node.determine_compatibility()
            if not curr_node.compatible:
                raise Error('Node is not compatible with the graph') 
        for curr_node in nodes:
            if curr_node not in self.node_to_int:
                self.max_no += 1
                self.adj_mat[self.max_no] = {}
                self.adj_list[self.max_no] = [[], []]
                self.node_to_int[curr_node] = self.max_no
                self.int_to_node[self.max_no] = curr_node
                self.nodes.append(self.max_no)
                if isinstance(curr_node, convolution_block):
                    self.conv_blocks.append(self.max_no)
                    self.node_dict[str(self.max_no)]=curr_node
                    
                elif isinstance(curr_node, max_pool_node):
                    self.max_pool_blocks.append(self.max_no)
                    self.node_dict[str(self.max_no)]=curr_node
        for curr_node in nodes:
            no = self.node_to_int[curr_node]
            try:
                self.adj_list[no] = [map(lambda x: self.node_to_int[x], curr_node.in_adj), map(lambda x: self.node_to_int[x], curr_node.out_adj)]
            except:
#                 print self.node_to_int
#                 print curr_node.in_adj
#                 print curr_node.out_adj
                raise Exception
            for par in self.adj_list[no][0]:
                self.adj_mat[par][no] = 1
            for child in self.adj_list[no][0]:
                self.adj_mat[no][child] = 1
        self.topsorting()
            
    def deepen_morph(self):
        deepen_conv_block = self.int_to_node[random.choice(self.conv_blocks)]
        kernel_size = random.choice([3, 5])
        in_channels, in_h, in_w = deepen_conv_block.output_shape
        out_channels = in_channels
        identity_conv_block = convolution_block(in_h, in_w, in_channels, out_channels, kernel_size, (kernel_size-1)/2)
        weights = identity_conv_block.conv_layer.weight.data
        
        # creating identity weights
        for channel in range(out_channels):
            for i in range(in_channels):
                for j in range(kernel_size):
                    for k in range(kernel_size):
                        weights[channel][i][j][k] = int((channel == i) and (j == k) and j == (kernel_size)/2 )
#         print 'weights of identity conv block', weights
        
        ## make connections 
        identity_conv_block.describe_adj_list([deepen_conv_block], deepen_conv_block.out_adj)
        deepen_conv_block.describe_adj_list(deepen_conv_block.in_adj, [identity_conv_block])

        #### later look at creating a function for singular change to in_adj or out_adj of nodes
        for out_node in identity_conv_block.out_adj:
            out_node_in_adj = [identity_conv_block if (x == deepen_conv_block) else x for x in out_node.in_adj ]
            out_node.describe_adj_list(out_node_in_adj, out_node.out_adj)
        
        self.add_nodes_to_network([deepen_conv_block, identity_conv_block] + identity_conv_block.out_adj)
    
    
    def widen_morph(self):
        candidate_conv_blocks = []
        for conv_block in self.conv_blocks:
            isCandidate = bool(len(self.adj_list[conv_block][1]))
            for child in self.adj_list[conv_block][1]:
                isCandidate = isCandidate and isinstance(self.int_to_node[child], convolution_block)
            if isCandidate:
                candidate_conv_blocks.append(conv_block)
        if len(candidate_conv_blocks) == 0:
            return False

        parent_block_no = random.choice(candidate_conv_blocks)
        parent_block = self.int_to_node[parent_block_no]
        widening_factor = random.choice([2, 4])
        in_channels, in_h, in_w = parent_block.input_shape
        out_channels = parent_block.out_channels
        kernel_size = parent_block.kernel_size
        padding = parent_block.padding
        stride = parent_block.stride
        widened_parent_block = convolution_block(in_h, in_w, in_channels, out_channels*widening_factor, kernel_size, padding, stride)
        original_parent_weight = parent_block.conv_layer.weight.data
        widened_parent_weight = widened_parent_block.conv_layer.weight.data
        widened_parent_weight[:out_channels] = original_parent_weight
        widened_parent_weight[out_channels:] = torch.zeros((out_channels*(widening_factor-1), in_channels, kernel_size, kernel_size))
        parent_block = widened_parent_block
#         self.int_to_node[parent_block_no]  = widened_parent_block
#         del self.node_to_int[parent_block]
#         self.node_to_int[widened_parent_block] = parent_block_no
        parent_out_adj = []
        for child in parent_block.out_adj:
            child_no = self.node_to_int[child]
            in_channels, in_h, in_w = child.input_shape
            out_channels = child.out_channels
            kernel_size = child.kernel_size 
            padding = child.padding
            stride = child.stride
            child_widened = convolution_block(in_h, in_w, in_channels*widening_factor, out_channels, kernel_size, padding, stride)
            child_widened.conv_layer.weight.data[:, :in_channels, :, :] = child.conv_layer.weight.data
#             child_widened.describe_adj_list([widened_parent_block if x == parent_block else x for x in child.in_adj], child.out_adj)
#             self.int_to_node[child_no] = child_widened
#             del self.node_to_int[child]
#             self.node_to_int[child_widened] = child_no
#             parent_out_adj.append(child_widened)
#         widened_parent_block.describe_adj_list(parent_block.in_adj, parent_out_adj)
    
    def dfs(self, curr_node, visited, weight):
        visited[curr_node] = weight
        for child in self.adj_list[curr_node][1]:
            if child not in visited:
                kernel = 0
                padding = 0
                constant = 0
                child_node = self.int_to_node[child]
                ## make adjustments for concatenation
                if isinstance(child_node, convolution_block) or isinstance(child_node, max_pool_node):
                    kernel = child_node.kernel_size
                    padding = child_node.padding
                    constant = 1
                self.dfs(child, visited, [weight[0]+kernel, weight[1]+padding, weight[2]+constant])
        
        
    def get_descendant_vectors(self):
        descs = {}
        for curr_node in self.nodes:
            visited = {}
            self.dfs(curr_node, visited, [0, 0, 0])
            del visited[curr_node] # remove root 
            descs[curr_node] = visited
        return descs
        
    def skip_morph(self):
        descs = self.get_descendant_vectors()
        candidates = [(ans, des) for ans in descs for des in descs[ans] ]
        no1, no2 = random.choice(candidates)
        weight = descs[no1][no2]
        #join outputs of node1 and node2 using a merge block
        node_a = self.int_to_node[no1]
        node_b = self.int_to_node[no2]
        out_ch_1, out_h_1, out_w_1 = node_a.output_shape
        out_ch_2, out_h_2, out_w_2 = node_b.output_shape
#         print 'selected_nodes are ', no1, "  ",no2
#         print 'weight is   ', weight
#         print(out_h_1, out_h_2, weight[0] - 2*weight[1] - weight[2])
        assert(out_h_1 - out_h_2 == out_w_1 - out_w_2)
        assert(out_h_1 - out_h_2 == weight[0] - 2*weight[1] - weight[2])

        if weight[2] & 1 == 0:
            weight[2] += 1
            weight[0] += 1
        weight[1] += (weight[2])/2
        weight[2] -= 2*(weight[2]/2)
        kernel_size = weight[0]
        padding = weight[1]
        stride = 1
        new_conv = convolution_block(out_h_1, out_w_1, out_ch_1, out_ch_2, kernel_size, padding, stride)
        new_add = add_node([new_conv, node_b], node_b.out_adj)
        new_conv.describe_adj_list([node_a], [new_add])
        new_conv.conv_layer.weight.data = torch.zeros(new_conv.conv_layer.weight.data.shape)
        node_a.describe_adj_list(node_a.in_adj, node_a.out_adj+[new_conv])
        for child_node in node_b.out_adj:
            child_node.describe_adj_list([new_add if x==node_b else x for x in child_node.in_adj], child_node.out_adj)
        node_b.describe_adj_list(node_b.in_adj, [new_add])
        self.add_nodes_to_network([node_a, node_b, new_conv, new_add] + new_add.out_adj)
        
        
        ###
    
    def visualize(self):
        graph = Digraph('./images/arch', './images/arch.gv')
        for no, curr_node in self.int_to_node.items():
#             graph.node(str(no), str(type(curr_node)).split('__main__.')[1])
            graph.node(str(no), str(self.node_to_int[curr_node]) + " :: " + repr(curr_node)[:200])
        for no, li in self.adj_list.items():
            for ch in li[1]:
                graph.edge(str(no), str(ch))
        graph.view()
#         x = IFrame("./images/archgv.pdf", width=600, height=300)
#         print x
    
    def describe(self):
        print 'Nodes: ', self.nodes
        print 'Conv_blocks', self.conv_blocks
        print 'Max_pool_blocks', self.max_pool_blocks
        print 'Adj_list', self.adj_list
        print 'Adj_mat', self.adj_mat
        print 'int_to_node', self.int_to_node
        print 'node_to_int', self.node_to_int
        print 'Toposort', self.topsort
        print 'node_dict', self.node_dict
    