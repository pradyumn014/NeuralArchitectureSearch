import torch.nn as nn
from node import  node

class fullyConnected(nn.Module,node) :
    def __init__(self, net) :   #nas
        super(fullyConnected,self).__init__()
        node.__init__(self, )
        self.net = net
        sh = net.int_to_node[net.topsort[-1]].output_shape
        out = sh[0]*sh[1]*sh[2]
#         out = nasout[1]*nasout[2]*nasout[3]
        self.fc1 = nn.Linear(out,16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16,10)
#         self.batch_size = nasout[0]
        
    def forward(self,x) :
        out = self.net(x)
#         out = out.view(self.batch_size,-1)
#         print(out.shape)
        out  =  out.view(out.shape[0], -1)
#         print 'out.shape', out.shape
#         [1 x 75264], m2: [2352 x 16]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

BATCH = 4
BEGIN_IN_CHANNELS = 3 
def addLinearLayers(net):
#     t = [32,3,28,28]
    net = fullyConnected(net)
    return net