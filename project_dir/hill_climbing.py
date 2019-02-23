from network import Network
from training import Train, Test
from nodes import *
import os
import pickle
import random

class hill_climbing(object):
    def __init__(self, iterations = 10, branching = 10, net = None):
        ## If init network is not provided, provide facility to begin with random network
        # X and Y are data and labels
        self.iterations = iterations 
        self.branching = branching
        if net:
            self.net = net
        else:
            n1 = convolution_block(3, 32, 32 , 32, 3)
            m1 = max_pool_block(32, 30, 30, 2)
            flatten = FlattenBlock((32, 29, 29))
            f1 = fullyConnectedBlock(32*29*29, 10)
            n1.describe_adj_list([], [m1])
            m1.describe_adj_list([n1], [flatten])
            flatten.describe_adj_list([m1], [f1])
            f1.describe_adj_list([flatten], [])
            self.net = Network({0:[[], [1]], 1:[[0], [2]], 2: [[1], [3]], 3:[[2], []]}, {0: n1, 1:m1, 2:flatten, 3:f1})
        # self.net.createModel()
    
    def plot_loss_graph(self):
        pass
    
    def visualize_tree_of_networks(self):
        pass
    
    def start(self):
        current_net = self.net
        path = '../assets/pickles/'

        for iter in range(self.iterations):
            print '*'*100
            print "Iteration: ", iter+1
            folder_path = os.path.join(path, '{}'.format(iter))
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            current_net.createModel()
            par_loss, _ = Train(current_net, 10, 128, 0.01, 0.001)
            par_path = os.path.join(folder_path, 'par')
            with open(par_path, 'wb') as f:
                pickle.dump(current_net, f)
            current_net.visualize(os.path.join(folder_path, 'image-par'))
            children = []
            no = 0
            while no < self.branching:
                print 'Child: ', no
                with open(par_path, 'rb') as f:
                    child = pickle.load(f)
                actions = {'deepen': child.deepen_morph, 
                'widen': child.widen_morph, 
                'skip': child.skip_morph }
                choice = random.choice(actions.keys())
                if not actions[choice]():
                    continue
                print 'Choice:', choice
                child_name = '{}-{}'.format(str(no), choice)
                child_path  = os.path.join(folder_path, child_name)
                child.visualize(os.path.join(folder_path, 'image-{}'.format(child_name)))
                child.createModel()
                loss, _ = Train(child, 10, 128, 0.01, 0.001)
                children.append((loss, child_name))
                with open(child_path, 'wb') as f:
                    pickle.dump(child, f)
                no = no + 1
            
            children.sort()
            best_child_name, best_child_loss = children[0]
            with open(os.path.join(folder_path, best_child_name), 'rb') as f:
                best_child = pickle.load(f)
            print "Iteration:", iter + 1
            print "ParentLoss: ",  par_loss
            print "BestChildLoss: ", best_child_loss
            current_net = best_child
        
        Test(current_net)
        




                