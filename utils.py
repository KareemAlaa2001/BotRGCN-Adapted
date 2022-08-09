import torch
from torch import nn
import wandb

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.kaiming_uniform_(m.weight)

def shuffleHomoGraph(des_tensor, tweets_tensor, \
            num_prop, category_prop, edge_index, edge_type, labels, train_idx):
    # shuffle the data
    perm = torch.randperm(len(train_idx))
    des_tensor[train_idx] = des_tensor[train_idx][perm]
    tweets_tensor[train_idx] = tweets_tensor[train_idx][perm]
    num_prop[train_idx] = num_prop[train_idx][perm]
    category_prop[train_idx] = category_prop[train_idx][perm]

    permDict = {og_index: new_index for og_index, new_index in zip(train_idx,perm)}

    # TODO this doesnt quite work but we go without it for now
    for i in range(len(edge_index)):
        for j in range(len(edge_index[i])):
            edge_index[i][j] = permDict[edge_index[i][j].item()]

    labels[train_idx] = labels[train_idx][perm]
    return des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels
        
