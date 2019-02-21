import torch
import torch.nn as nn
from nodes import *

import numpy as np
import random
from graphviz import Digraph

seed = 53
np.random.seed(seed)
torch.manual_seed(seed)

class Network(nn.Module, object):
    def __init__(self):
        super(Network, self).__init__()
        self.adj_mat = {}
        self.adj_list = {}
        self.int_to_node = {}
        self.node_to_int = {}
        self.nodes = []
        self.conv_blocks = []
        self.max_pool_blocks = [] 
        self.topsort = []
        self.rank_in_topsort = {}
        self.max_no = 0
        self.node_dict = {}
        self.image_path = './assets/images/net' 

        
    def __init__(self, adj_list, int_to_node, filename  = 'net'):
        super(Network, self).__init__()        
#         assert len(adj_list) == assert(int_to_node)
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
        self.image_path =  './assets/images/' + filename

        
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
    
    def get_node_to_int(self, int_to_node):
        node_to_int = {}
        for no, cnode in int_to_node.items():
            node_to_int[cnode] = no
        return node_to_int

    def topsorting(self):
        # level problem
        topsort = []
        import Queue
        in_deg = {}
        q = Queue.Queue()
        for node in self.nodes:
            val  = len(self.adj_list[node][0])
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
        
    def createModel(self):
         self.node_dict = torch.nn.ModuleDict(self.node_dict)
        
    def forward(self, x):
        self.topsorting() # though this is not required here
        outputs = {}
        outputs[self.topsort[0]] = self.int_to_node[self.topsort[0]].forward(x)
        for ind in range(1, len(self.topsort)):
            node_no = self.topsort[ind]
            curr_node = self.int_to_node[node_no]
            outputs[node_no] = curr_node.forward(*map(lambda x: outputs[x], self.adj_list[node_no][0]))
        return outputs[self.topsort[-1]]
    
    def get_conv_and_max_pool_blocks(self):
        conv_blocks = []
        max_pool_blocks = []
        for x in self.nodes:
            if isinstance(self.int_to_node[x], convolution_block):
                conv_blocks.append(x)
                self.node_dict[str(x)]=self.int_to_node[x]
            elif isinstance(self.int_to_node[x], max_pool_block):
                max_pool_blocks.append(x)
                self.node_dict[str(x)]=self.int_to_node[x]
        return (conv_blocks, max_pool_blocks)
        
 
    def add_nodes_to_network(self, nodes):
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
                elif isinstance(curr_node, max_pool_block):
                    self.max_pool_blocks.append(self.max_no)
                    self.node_dict[str(self.max_no)]=curr_node
                    
        for curr_node in nodes:
            no = self.node_to_int[curr_node]
            self.adj_list[no] = [map(lambda x: self.node_to_int[x], curr_node.in_adj), map(lambda x: self.node_to_int[x], curr_node.out_adj)]
            for child in curr_node.out_adj:
                self.adj_mat[no][self.node_to_int[child]] = 1
#                 self.adj_list[self.node_to_int[child]][0].append(no)
        self.topsorting()
    
    
    def remove():
        pass
    
    def deepen_morph(self):
        deepen_conv_block = self.int_to_node[random.choice(self.conv_blocks)]
        kernel_size = random.choice([3, 5])
        in_channels, in_h, in_w = deepen_conv_block.output_shape
        out_channels = in_channels
        identity_conv_block = convolution_block(in_channels, in_h, in_w, out_channels, kernel_size, (kernel_size-1)/2)
#         weights = identity_conv_block.conv_layer.weight.data
        weights = identity_conv_block.conv_layer.weight
        
        # creating identity weights
        for channel in range(out_channels):
            for i in range(in_channels):
                for j in range(kernel_size):
                    for k in range(kernel_size):
                        weights[channel][i][j][k] = int((channel == i) and (j == k) and j == (kernel_size)/2 )
        
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

#         parent_block_no = random.choice(candidate_conv_blocks)
        parent_block = self.int_to_node[random.choice(candidate_conv_blocks)]
        widening_factor = random.choice([2, 4])
        in_channels, in_h, in_w = parent_block.input_shape
        out_channels = parent_block.out_channels
        kernel_size = parent_block.kernel_size
        padding = parent_block.padding
        stride = parent_block.stride
        original_parent_weight = parent_block.conv_layer.weight
        #set_params
        parent_block.refresh(in_channels, in_h, in_w, out_channels*widening_factor, kernel_size, padding, stride)
        widened_parent_weight = parent_block.conv_layer.weight

        widened_parent_weight[:out_channels] =torch.nn.Parameter(original_parent_weight)
        widened_parent_weight[out_channels:] = torch.nn.Parameter(torch.zeros((out_channels*(widening_factor-1), in_channels, kernel_size, kernel_size)))

        for child in parent_block.out_adj:
            child_no = self.node_to_int[child]
            in_channels, in_h, in_w = child.input_shape
            out_channels = child.out_channels
            kernel_size = child.kernel_size 
            padding = child.padding
            stride = child.stride
            original_child_weight = child.conv_layer.weight
            child.refresh(in_channels*widening_factor, in_h, in_w, out_channels, kernel_size, padding, stride)
            child.conv_layer.weight[:, :in_channels, :, :] = torch.nn.Parameter(original_child_weight)

        
    ## Start from here
    def dfs(self, curr_node, visited, weight):
        curr_node_obj = self.int_to_node[curr_node]
        if isinstance(curr_node_obj, FlattenBlock) or isinstance(curr_node_obj, fullyConnectedBlock):
            return
        visited[curr_node] = weight
        for child in self.adj_list[curr_node][1]:
            if child not in visited:
                kernel = 0
                padding = 0
                constant = 0
                child_node = self.int_to_node[child]
                ## make adjustments for concatenation
                if isinstance(child_node, convolution_block) or isinstance(child_node, max_pool_block):
                    kernel = child_node.kernel_size
                    padding = child_node.padding
                    constant = 1
                self.dfs(child, visited, [weight[0]+kernel, weight[1]+padding, weight[2]+constant])
        
        
    def get_descendant_vectors(self):
        descs = {}
        for curr_node in self.nodes:
            if isinstance(self.int_to_node[curr_node], FlattenBlock) or isinstance(self.int_to_node[curr_node], fullyConnectedBlock):
                continue
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
#         print('Skip Morphism', type(node_a), type(node_b))
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
        stride = weight[2]
        assert stride == 1
        new_conv = convolution_block(out_ch_1, out_h_1, out_w_1, out_ch_2, kernel_size, padding, stride)
        new_add = add_block([new_conv, node_b], node_b.out_adj)
        new_conv.describe_adj_list([node_a], [new_add])
        new_conv.conv_layer.weight = torch.nn.Parameter(torch.zeros(new_conv.conv_layer.weight.data.shape))
        node_a.describe_adj_list(node_a.in_adj, node_a.out_adj+[new_conv])
        for child_node in node_b.out_adj:
            child_node.describe_adj_list([new_add if x==node_b else x for x in child_node.in_adj], child_node.out_adj)
        node_b.describe_adj_list(node_b.in_adj, [new_add])
        self.add_nodes_to_network([node_a, node_b, new_conv, new_add] + new_add.out_adj)
        
        
    def visualize(self):
        graph = Digraph(self.image_path, self.image_path+'.gv')
        for no, curr_node in self.int_to_node.items():
#             graph.node(str(no), str(type(curr_node)).split('__main__.')[1])
            graph.node(str(no), str(no) + " :: " + repr(curr_node))
        for no, li in self.adj_list.items():
            for ch in li[1]:
                graph.edge(str(no), str(ch))
        graph.view()

    
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
    