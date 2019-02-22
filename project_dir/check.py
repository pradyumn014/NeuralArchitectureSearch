from network import Network
from nodes import *
import random
from training import Train, Test

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

def dummy_train():    
    n1 = convolution_block(3, 32, 32 , 32, 3)
    n2 = convolution_block(32, 30, 30, 32, 3)
    m1 = max_pool_block(32, 28, 28, 2)
    n3 = convolution_block(32, 27, 27 , 64, 3)
    n4 = convolution_block(64, 25, 25 , 64, 3)
    m2 = max_pool_block(64, 23, 23, 2)

    flatten = FlattenBlock((64, 22, 22))
    f1 = fullyConnectedBlock(64*22*22, 32, True, 'relu')
    f2 = fullyConnectedBlock(32, 10, True, 'relu')
    
    n1.describe_adj_list([], [n2])
    n2.describe_adj_list([n1], [m1])
    m1.describe_adj_list([n2], [n3])
    n3.describe_adj_list([m1], [n4])
    n4.describe_adj_list([n3], [m2])
    m2.describe_adj_list([n4], [flatten])
    flatten.describe_adj_list([m2], [f1])
    f1.describe_adj_list([f1], [f2])
    f2.describe_adj_list([f1], [])
    

    net = Network({0:[[], [1]], 1:[[0], [2]], 2: [[1], [3]], 3:[[2], [4]], 4:[[3], [5]], 5:[[4], [6]], 6:[[5], [7]], 7:[[6], [8]], 8:[[7], []]}, {0: n1, 1:n2, 2:m1, 3:n3, 4:n4, 5:m2, 6:flatten, 7:f1, 8:f2}, 'test_net')
    net.createModel()
    Train(net, 1, 0.01, 0.001)
    Test(net)

def dummy_train1():    
    n1 = convolution_block(3, 32, 32 , 32, 3)
    n2 = convolution_block(32, 30, 30, 32, 3)
    m1 = max_pool_block(32, 28, 28, 2)
    flatten = FlattenBlock((32, 27, 27))
    f1 = fullyConnectedBlock(32*27*27, 10)
    # f2 = fullyConnectedBlock(32, 10)

    n1.describe_adj_list([], [n2])
    n2.describe_adj_list([n1], [m1])
    m1.describe_adj_list([n2], [flatten])
    # n3.describe_adj_list([m1], [n4])
    # n4.describe_adj_list([n3], [m2])
    # m2.describe_adj_list([n4], [flatten])
    flatten.describe_adj_list([m1], [f1])
    f1.describe_adj_list([f1], [])
    # f2.describe_adj_list([f1], [])


    net = Network({0:[[], [1]], 1:[[0], [2]], 2: [[1], [3]], 3:[[2], [4]], 4:[[3], []]}, {0: n1, 1:n2, 2:m1, 3:flatten, 4:f1}, 'test_net')
    net.createModel()
    Train(net, 1, 128,  0.01, 0.001)
    Test(net)   


if  __name__ == "__main__":
    # graph_test()
    dummy_train1()
