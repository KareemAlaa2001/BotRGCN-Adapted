from HeteroTwibot import HeteroTwibot, initializeHeteroTwibot
from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero

from torch_geometric.loader import DataLoader, NeighborLoader

from augmodels import TweetAugmentedHAN, TweetAugmentedRGCN, TweetAugmentedHAN2, TweetAugmentedHAN2ExtraLayer, TweetAugHetGCN, TweetAugHANConfigurable
from model import BotRGCN

import torch
from tqdm import tqdm

import torch
from tqdm import tqdm

from torch import nn, svd
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score, precision_recall_curve, confusion_matrix
import wandb

print("we in trainTestHetero.py")

def trainTestHetero(embedding_size = 128, dropout = 0.3, lr = 1e-3, weight_decay = 5e-3, svdComponents = 100, thirds = False, epochs = 100, extraLayer=True, numHanLayers = 2):
    wandb.init(project="test-project", entity="graphbois")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## IMPORTING THE DATASET
    print("importing the dataset...")

    dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
    dataset = initializeHeteroTwibot(dataset)

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
        train(epoch,model, dataset, loss, optimizer)
        
    acc_test,loss_test,f1, roc_auc = test(model, dataset, loss)

    return roc_auc

# trainTestHetero()


# ## IMPORTING THE DATASET
# print("importing the dataset...")

# dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset = initializeHeteroTwibot(dataset)

# model = TweetAugmentedHAN2(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents, metadata=dataset.metadata()).to(device)

# wandb.config.update({
#   "model_name": model.__class__.__name__,
#   "dataset": dataset.__class__.__name__,
#   "embedding_size": embedding_size,
#   "dropout": dropout,
#   "lr": lr,
#   "weight_decay": weight_decay, 
#   "svdComponents": svdComponents, 
#   "thirds": thirds,
#   "epochs": epochs
# })

# # wandb.watch(model)

# loss=nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(),
#                     lr=lr,weight_decay=weight_decay)



# model.apply(init_weights)

# print("beginning training...")

# for epoch in tqdm(range(epochs), miniters=5):
#     train(epoch)
    
# test()

def train(epoch, model, dataset, loss, optimizer):
    model.train()
    output = model(dataset)
    loss_train = loss(output[dataset['user'].train_idx], dataset['user'].y[dataset['user'].train_idx])
    acc_train = accuracy(output[dataset['user'].train_idx],  dataset['user'].y[dataset['user'].train_idx])
    acc_val = accuracy(output[dataset['user'].val_idx], dataset['user'].y[dataset['user'].val_idx])
    loss_val = loss(output[dataset['user'].val_idx], dataset['user'].y[dataset['user'].val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    wandb.log({"loss_train": loss_train, "acc_train": acc_train, "acc_val": acc_val, "loss_val": loss_val})

    if (epoch+1)  % 5 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'loss_val: {:.4f}'.format(loss_val.item()))
    return acc_train,loss_train

def test(model, dataset, loss):
    model.eval()
    output = model(dataset)
    labels_test = dataset['user'].y[dataset['user'].test_idx]
    output_test = output[dataset['user'].test_idx]
    loss_test = loss(output[dataset['user'].test_idx], dataset['user'].y[dataset['user'].test_idx])
    acc_test = accuracy(output[dataset['user'].test_idx], dataset['user'].y[dataset['user'].test_idx])
    output = output.max(1)[1].to('cpu').detach().numpy()
    label = dataset['user'].y.to('cpu').detach().numpy()
    f1 = f1_score( dataset['user'].y[dataset['user'].test_idx],output[dataset['user'].test_idx])
    mcc = matthews_corrcoef( dataset['user'].y[dataset['user'].test_idx],output[dataset['user'].test_idx])
    prec = precision_score( dataset['user'].y[dataset['user'].test_idx], output[dataset['user'].test_idx])
    recall = recall_score( dataset['user'].y[dataset['user'].test_idx], output[dataset['user'].test_idx])
    roc_auc = roc_auc_score( dataset['user'].y[dataset['user'].test_idx], output[dataset['user'].test_idx])
    conf = confusion_matrix(dataset['user'].y[dataset['user'].test_idx], output[dataset['user'].test_idx])
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "f1_score= {:.4f}".format(f1.item()),
            "mcc= {:.4f}".format(mcc.item()),
            )

    wandb.log({"loss_test": loss_test, "acc_test": acc_test, "f1_score": f1, "mcc": mcc, "prec": prec, "recall": recall, "roc_auc": roc_auc, "confusion_matrix": conf})

    # Optional
    
    return acc_test,loss_test,f1, roc_auc



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
    lr = 5e-3
    weight_decay = 5e-3
    svdComponents = 200
    thirds = True
    epochs = 30
    extraLayer = True
    numHanLayers = 2


    trainTestHetero(embedding_size, dropout, lr, weight_decay, svdComponents, thirds, epochs, extraLayer, numHanLayers)
