from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import random
import get_targets as tp



device = torch.device("cpu")

class Net(nn.Module):

    def __init__(self, n_in, n_h, n_h2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc2 = nn.Linear(n_h, n_h2)
        #self.fc3 = nn.Linear(n_h2, n_h3)
        #self.fc4 = nn.Linear(n_h3, n_out)

    def forward(self, x):
        out1 = torch.sigmoid(self.fc1(x))
        out2 = torch.sigmoid(self.fc2(out1))
        #out3 = torch.sigmoid(self.fc3(out2))
        #out4 = torch.sigmoid(self.fc4(out3))
        return [x, out1, out2]

model = Net(784,512,10).to(device)
model.load_state_dict(torch.load("state_dict_model.pt"))
model.eval()


#####Features Extration#########

#forward with vector zero
#images = torch.zeros([784]).requires_grad_(True)

activations = model(torch.zeros([784]).requires_grad_(True))

#####Compute Targets from all layers#######
targets = {}

#output one-hot target
t = []
for i in range(10):
    t.append(torch.zeros(10))
    t[i].to(device)
    t[i][i] = 1.0

features = []
for i in t:

    targets, _ = tp.get_targets(activations, i, model)
    features.append(targets[activations[0]])

#Print input neurons expected 
fig, ax = plt.subplots(nrows=2, ncols=5)
flag = 0
for i in range(2):
    for j in range(5):
        print(flag)
        ax[i, j].imshow(features[flag].reshape(28,28))
        #ax[i, j].set_title('Neuron {}'.format(flag),fontsize=10)
        ax[i, j].axis('off')
        flag += 1

#fig.suptitle('{}'.format(w_path), y=0.98 ,fontsize=16)
#fig.suptitle('{}'.format(w_path), y=0.98 ,fontsize=16)
#fig.savefig("../features_hidden/{}.png".format(w_path), bbox_inches='tight')
plt.show()

