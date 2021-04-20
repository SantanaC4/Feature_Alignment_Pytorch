import torch
import numpy as np 

def get_targets(target, a, model):

    ####extraction feature target#######
    targets = {}
    targets[a[-1]] = target


    #Unwrap model 
    def unwrap_model(model):
        weights = []
        for name, weight in model._modules.items():
            weights.append(weight)

        return weights

    #Get Weights 
    weights = unwrap_model(model)

    taus = []
    for weight in weights:
        tau = []
        for row in weight.weight.T:
            t = (1/torch.sum(row**2))
            tau.append(torch.maximum(t,torch.tensor(1)))

        taus.append(torch.tensor(tau).detach())

    loss = 0.5*torch.sum((targets[a[-1]] - a[-1])**2.0)

    #########Feature Alignment############
    for i in range(len(a)-2,-1,-1):

        targets[a[i]] = a[i] - taus[i]*(torch.autograd.grad(loss, a[i], retain_graph=True)[0])
        loss = 0.5*torch.sum((targets[a[i]]-a[i])**2.0)
        #targets[a[i]] = targets[a[i]].detach()
    ######################################

    return targets, targets[a[0]]

def get_targets2(target, a, model):

    targets = {}
    targets[a[-1]] = target


    #Unwrap model 
    def unwrap_model(model):
        weights = []
        for name, weight in model._modules.items():
            weights.append(weight)

        return weights

    #Get Weights 
    weights = unwrap_model(model)

    taus = []
    for weight in weights:
        tau = []
        for row in weight.weight.T:
            t = (1/torch.sum(row**2))
            tau.append(torch.maximum(t,torch.tensor(1)))

        taus.append(torch.tensor(tau).detach())

    #######extraction feature ##########
    loss = 0.5*torch.sum((targets[a[-1]] - a[-1])**2.0)

    for i in range(len(a)-2,-1,-1):

        targets[a[i]] = a[i] - taus[i]*(torch.autograd.grad(loss, a[i], retain_graph=True)[0])
        loss = 0.5*torch.sum((targets[a[i]]-a[i])**2.0)
        targets[a[i]] = targets[a[i]].detach()
    ####################################

    return targets, targets[a[0]]

