from matplotlib import pyplot as plt
import torch
from pyunlocbox import functions
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import get_targets as tp

#Check Device configuration
#device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#Check Device configuration
#device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")


#Total Variation 
#f_tv = functions.norm_tv(maxit=50,dim=2)

def total_variation_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


class Net(nn.Module):

    def __init__(self, n_in, n_h, n_h2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc2 = nn.Linear(n_h, n_h2)
        self.d = nn.Linear(784,784)
        #self.fc3 = nn.Linear(n_h2, n_h3)
        #self.fc4 = nn.Linear(n_h3, n_out)

    def forward(self, x):
        out1 = torch.sigmoid(self.fc1(x))
        out2 = torch.sigmoid(self.fc2(out1))
        #out3 = torch.sigmoid(self.fc3(out2))
        #out4 = torch.sigmoid(self.fc4(out3))
        return [x, out1, out2]

    def decoder(self, x):
       return torch.sigmoid(self.d(x))


model = Net(784,512,120).to(device)

#Xavier initialization4
with torch.no_grad():
    for p in model.parameters():
        if len(p.shape) == 2:
            n = p.shape[1]
            s = (1.0/n)**0.5
            p.data = torch.normal(0.0,s,p.shape,device=device)

#random seed
manualSeed = 1
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#Hyperparameter 
batch_size = 50
lr = 0.001
num_epochs = 1
eps = 0.01
##Function to print features##
def matplot_print(features):
    fig, ax = plt.subplots(nrows=10, ncols=5)
    flag = 0
    for i in range(10):
        for j in range(5):
            ax[i, j].imshow(features[flag].detach().reshape(28,28))
            #ax[i, j].set_title('Neuron {}'.format(flag),fontsize=10)
            ax[i, j].axis('off')
            flag += 1

    print(flag)
    plt.show()
    plt.close()

## Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#loss_fn = torch.nn.MSELoss(reduction='sum')

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):

        #Flatten the image from  28*28 to 784 column vector
        #inputs.resize_((batch_size,1,784))
        inputs = inputs.view(inputs.shape[0], -1)

        inputs = inputs.to(device)
        #inputs.requires_grad_(True)
        labels = labels.to(device)
        outputs = model(inputs)

        B = len(inputs)
        target = (torch.randn(B, 10)).to(device)
        target.fill_(0.0 + eps)

        for example in range(B):
            target[example][labels[example]] = 1.0 - eps

        ##Get Targets 
        #_, feature = tp.get_targets(model(inputs), model(torch.rand(inputs.shape).requires_grad_(True)), model)

        #######Feature Alignment############
        output = model(inputs)[-1]
        u0 = 0*torch.rand(inputs.shape).requires_grad_(True)
        f_u0 = model(u0)[-1]

        loss = 0.5*torch.sum((output - f_u0)**2)+total_variation_loss(u0.reshape((len(u0),1,28,28)), 5)
 
        print(loss)
        u = u0 - torch.autograd.grad(loss, u0, retain_graph=True, create_graph=True)[0]
        u = model.decoder(u)

        #_, feature = tp.get_targets(output, model(torch.rand(inputs.shape).requires_grad_(True)), model)
        #matplot_print(u)
        RL = 0.5*torch.sum((inputs - u)**2)
        #+ torch.sum(u**6)
        ####################################


        loss = RL
        #loss = 0.5*torch.sum((inputs-feature))**2.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #_, feature = tp.get_targets(output, model(torch.rand(inputs.shape).requires_grad_(True)), model)
    #print(outputs.size())

    _, feature = tp.get_targets2(output, model(torch.rand(inputs.shape).requires_grad_(True)), model)
    matplot_print(feature)

    # Test model 
    #model.eval()
    #with torch.no_grad():
    #    correct = 0
    #    total = 0
    #    loss_test = 0.0
    #    for inputs, labels in test_loader:
    #        inputs = inputs.to(device)
    #        labels = labels.to(device)
    #        inputs = inputs.view(inputs.shape[0], -1)
    #        outputs = model(inputs)[-1]

    #        B = len(inputs)
    #        target = (torch.randn(B, 10)).to(device)
    #        target.fill_(0.0 + eps)

    #        for example in range(B):
    #            target[example][labels[example]] = 1.0 - eps

    #        loss_test += (0.5*torch.sum((target-outputs)**2.0)).item()

    #        _, predicted = torch.max(outputs.data, 1)
    #        total += labels.size(0)

    #        correct += (predicted == labels).sum().item()
    #    print('Test Accuracy of the model on the 10000 test inputs: {} %'.format(100*correct / total))


#Save state_dict 
PATH = "state_dict_model.pt"
torch.save(model.state_dict(),PATH)

