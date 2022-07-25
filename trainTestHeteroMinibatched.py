from HeteroTwibot import HeteroTwibot, initializeHeteroTwibot
from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero

from torch_geometric.loader import DataLoader, NeighborLoader

from augmodels import TweetAugmentedHAN, TweetAugmentedRGCN, TweetAugmentedHAN2, TweetAugmentedHAN2ExtraLayer, TweetAugHetGCN, TweetAugHANConfigurable
from model import BotRGCN
from trainTestHetero import test

import torch
from tqdm import tqdm

from torch import nn, svd
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score, precision_recall_curve, confusion_matrix
import wandb


def train_minibatched(epoch, model, train_loader, loss, optimizer, device):
    model.train()

    total_examples = total_loss = total_acc = 0
    
    for data in train_loader:
        data = data.to(device, 'edge_index')

        batch_size = data['user'].batch_size
        output = model(data)

        loss_batch = loss(output[:batch_size], data['user'].y[:batch_size])
        acc_batch = accuracy(output[:batch_size],  data['user'].y[:batch_size])

        # acc_val = accuracy(output[data['user'].val_idx], data['user'].y[data['user'].val_idx])
        # loss_val = loss(output[data['user'].val_idx], data['user'].y[data['user'].val_idx])

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += loss_batch * batch_size
        total_acc += acc_batch * batch_size

    
    train_loss = total_loss / total_examples 
    train_acc = total_acc / total_examples

    return train_loss, train_acc

def test_minibatched(loader, model, loss, device):
    model.eval()
    total_examples = total_loss = total_acc = 0
    for data in loader:
        data = data.to(device, 'edge_index')

        batch_size = data['user'].batch_size
        output = model(data)

        loss_batch = loss(output[:batch_size], data['user'].y[:batch_size])
        acc_batch = accuracy(output[:batch_size],  data['user'].y[:batch_size])

        total_examples += batch_size
        total_loss += loss_batch * batch_size
        total_acc += acc_batch * batch_size

    test_loss = total_loss / total_examples 
    test_acc = total_acc / total_examples

    return test_loss, test_acc


def trainTestHeteroMinibatched(embedding_size = 128, dropout = 0.3, lr = 1e-3, weight_decay = 5e-3, svdComponents = 100, thirds = False, epochs = 100, extraLayer=True, numHanLayers = 2, neighboursPerNode = 50, batch_size = 256):
    wandb.init(project="test-project", entity="graphbois")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## IMPORTING THE DATASET
    print("importing the dataset...")

    dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
    dataset = initializeHeteroTwibot(dataset)

    kwargs = {'num_workers': torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1, 'persistent_workers': True, 'batch_size': batch_size}

    train_loader = NeighborLoader(dataset, num_neighbors=[neighboursPerNode] * numHanLayers,shuffle=False, input_nodes=('user',dataset['user'].train_mask), **kwargs)
    val_loader = NeighborLoader(dataset, num_neighbors=[neighboursPerNode] * numHanLayers,shuffle=False, input_nodes=('user',dataset['user'].val_mask), **kwargs)



    # model = TweetAugmentedHAN2ExtraLayer(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents, metadata=dataset.metadata()).to(device)
    model = TweetAugHANConfigurable(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents, metadata=dataset.metadata(), extraLayer=extraLayer,numHanLayers=numHanLayers).to(device)
    wandb.config.update({
    "model_name": model.__class__.__name__,
    "dataset": dataset.__class__.__name__,
    "embedding_size": embedding_size,
    "dropout": dropout,
    "lr": lr,
    "weight_decay": weight_decay, 
    "svdComponents": svdComponents, 
    "thirds": thirds,
    "epochs": epochs
    })

    if  model.__class__.__name__ == "TweetAugHANConfigurable":
        wandb.config.update({
        "extraLayer": extraLayer,
        "numHanLayers": numHanLayers
        })
    # wandb.watch(model)

    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                        lr=lr,weight_decay=weight_decay)

    model.apply(init_weights)

    print("beginning training...")

    for epoch in tqdm(range(epochs), miniters=5): 
        train_loss, train_acc = train_minibatched(epoch,model, train_loader, loss, optimizer, device)
        val_loss, val_acc = test_minibatched(val_loader, model, loss, device)
        wandb.log({"loss_train": train_loss, "acc_train": train_acc, "acc_val": val_acc, "loss_val": val_loss})

        if (epoch+1)  % 5 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(train_loss.item()),
                'acc_train: {:.4f}'.format(train_acc.item()),
                'acc_val: {:.4f}'.format(val_loss.item()),
                'loss_val: {:.4f}'.format(val_acc.item()))

    acc_test,loss_test,f1, roc_auc = test(model, dataset, loss)

    return roc_auc

if __name__ == '__main__':
    wandb.init(project="test-project", entity="graphbois")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## HYPERPARAMETERS
    # TODO implement an arg_parser module for the hyperparameters

    # Default values
    # embedding_size,dropout,lr,weight_decay, svdComponents, thirds=128,0.3,1e-3,5e-3, 100, False

    # Current Values
    embedding_size = 128
    dropout = 0.5
    lr = 1e-3
    weight_decay = 5e-3
    svdComponents = 200
    thirds = True
    epochs = 50
    extraLayer = True
    numHanLayers = 4
    neighboursPerNode = 200
    batch_size = 1024

    trainTestHeteroMinibatched(embedding_size, dropout, lr, weight_decay, svdComponents, thirds, epochs, extraLayer, numHanLayers, neighboursPerNode, batch_size)