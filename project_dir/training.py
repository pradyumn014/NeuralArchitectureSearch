import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import time


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='../assets/data/cifardata', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='../assets/data/cifardata', train=False, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)


def Test(net) :
    net.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available() :
                images, labels = images.cuda() , labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    total  = 0
    correct = 0
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        total += class_total[i]
        correct += class_correct[i]
    print((correct/total)*100)

def Train(net, epochs, lr_start, lr_end):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    training_start_time = time.time()
	# use lr_start, lr_end for annealing 
	# use SGDR instead of SGD : experiment with other optims
    print 'Train Data', len(trainloader)*4
    for epoch in range(epochs):  # loop over the dataset multiple times
        # break
        running_loss = 0.0
        n_batches = len(trainloader)
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            # print len(trainloader)

            # get the inputs
            
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

	        # if torch.cuda.is_available() :
	        # 	inputs, labels = inputs.cuda() , labels.cuda()
	        # zero the parameter gradients

            optimizer.zero_grad()

	        # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            #for accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()  

	        # print statistics
            running_loss += loss.item()
            total_train_loss += loss.data[0]

            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s Training Accuracy: {:.2f} ".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time, 100.*correct/total))
                
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

	        # viewerval = 100

	        # print 'Done [{}|{}]'.format(i,running_loss)


	        # if i % viewerval == viewerval-1:    # print every 2000 mini-batches
	        #     print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / viewerval))
	        #     running_loss = 0.0

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

