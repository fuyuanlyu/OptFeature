import sys, os
import torch
import pickle
from dataloader import tfloader
import modules.models as models
from modules.optfeature import OptFeature

def getModel(model:str, opt, device):
    model = model.lower()
    if model == "lr":
        return models.LR(opt).to(device)
    elif model == "fm":
        return models.FM(opt).to(device)
    elif model == "deepfm":
        return models.DeepFM(opt).to(device)
    elif model == "fnn":
        return models.FNN(opt).to(device)
    elif model == "dcn":
        return models.DeepCrossNet(opt).to(device)
    elif model == "ipnn":
        return models.InnerProductNet(opt).to(device)
    elif model == "optfeature":
        return OptFeature(opt, device).to(device)
    else:
        raise ValueError("Invalid model type: {}".format(model))

def getOptimOne(network, optim, lr, l2):
    params = network.parameters()
    optim = optim.lower()
    if optim == "sgd":
        return torch.optim.SGD(params, lr = lr, weight_decay = l2)
    elif optim == "adam":
        return torch.optim.Adam(params, lr = lr, weight_decay = l2)
    else:
        raise ValueError("Invalid optmizer type:{}".format(optim))
    
def getOptim(network, optim, lr, l2, fi_lr, fi_l2):
    optim = optim.lower()

    network_params, fi_params, alpha_params = [], [], []
    network_names, fi_names, alpha_names = [], [], []
    for name, param in network.named_parameters():
        if "alpha" in name:
            alpha_params.append(param)
            alpha_names.append(name)
        elif "FI" in name:
            fi_params.append(param)
            fi_names.append(name)
        else:
            network_params.append(param)
            network_names.append(name)
            
    print("_"*30)
    print("alpha_names", alpha_names)
    print("_"*30)
    print("fi_names:", fi_names)
    print("_"*30)
    print("network_names:", network_names)
    print("_"*30)

    alpha_group = {
        "params": alpha_params,
        'lr': 1e-3
    }
    fi_group = {
        'params': fi_params,
        'weight_decay': fi_l2,
        'lr': fi_lr
    }
    network_group = {
        'params': network_params,
        'weight_decay': l2,
        'lr': lr
    }
    if optim == 'sgd':
        optimizer = torch.optim.SGD([network_group])
        fi_optimizer = torch.optim.SGD([fi_group, alpha_group])
    elif optim == 'adam':
        optimizer = torch.optim.Adam([network_group])
        fi_optimizer = torch.optim.Adam([fi_group, alpha_group])
    else:
        print("Optimizer not supported.")
        sys.exit(-1)

    return [optimizer, fi_optimizer]

def getDevice(device_id):
    if device_id != -1:
        assert torch.cuda.is_available(), "CUDA is not available"
        # torch.cuda.set_device(device_id)
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def getDataLoader(dataset:str, path):
    dataset = dataset.lower()
    if dataset == 'criteo':
        return tfloader.CriteoLoader(path)
    elif dataset == 'avazu':
        return tfloader.Avazuloader(path)
    elif dataset == 'kdd12':
        return tfloader.KDD12loader(path)
