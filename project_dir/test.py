from network import Network
from nodes import *
import random

def graph_test():    
    n1 = convolution_block(3, 32, 32 , 4, 3)
    n2 = convolution_block(4, 30, 30, 3, 3)
    flatten = FlattenBlock((3, 28, 28))
    f1 = fullyConnectedBlock(2352, 64, True)
    f2 = fullyConnectedBlock(64, 32)
    
    n1.describe_adj_list([], [n2])
    n2.describe_adj_list([n1], [flatten])
    flatten.describe_adj_list([n2], [f1])
    f1.describe_adj_list([f1], [f2])
    f2.describe_adj_list([f1], [])
    

    net = Network({0:[[], [1]], 1:[[0], [2]], 2: [[1], [3]], 3:[[2], [4]], 4:[[3], []]}, {0: n1, 1:n2, 2:flatten, 3:f1, 4:f2}, 'test_net')
#     net.describe() 
    net.visualize()
    for nan in net.nodes:
        print net.int_to_node[nan].input_shape, ' -> ', nan, '->', net.int_to_node[nan].output_shape 
    for i in range(20):
        operations = [net.skip_morph, net.deepen_morph, net.widen_morph]
        op = random.choice(operations)
        print "Iteration", i, op
        op()
    net.visualize()

if  __name__ == "__main__":
    graph_test()