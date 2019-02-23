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


#Training
n_training_samples = 40000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

#Validation
n_val_samples = 10000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

#Test
n_test_samples = 10000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))


def get_train_loader(batch_size):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    return trainloader

# trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)


def Test(net) :
    print "Testing"
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


def Train(net, epochs, batch_size, lr_start, lr_end):
    trainloader = get_train_loader(batch_size)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    training_start_time = time.time()
	# use lr_start, lr_end for annealing 
	# use SGDR instead of SGD : experiment with other optims
    # print 'Train Data Size', len(trainloader)*batch_size

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        n_batches = len(trainloader)
        print_every = n_batches
        start_time = time.time()
        total_train_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
 
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
            # total_train_loss += loss.data[0]
            total_train_loss += loss.data.item()

            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s Training Accuracy: {:.2f} ".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time, 100.*correct/total))
                
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        total_val_loss = 0
        val_total = 0
        val_correct = 0
        for inputs, labels in val_loader:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)

            #for accuracy
            _, predicted = val_outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item() 

            val_loss_size = criterion(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()
            
        print("Epoch {}, Validation loss = {:.2f}".format(epoch+1, total_val_loss / len(val_loader)))
        print('Epoch{}, Accuracy {}'.format(epoch+1, str(100.*val_correct/val_total)))
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return (total_val_loss / len(val_loader), val_correct/val_total)

