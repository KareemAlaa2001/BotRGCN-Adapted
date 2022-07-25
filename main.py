from model import BotRGCN
from augmodels import TweetAugmentedRGCN, TweetAugmentedHAN
# from Dataset import Twibot20
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD
from TwibotSmallAugmentedTSVDHomogeneous import TwibotSmallAugmentedTSVDHomogeneous
from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero
from HeteroTwibot import HeteroTwibot


import torch
from tqdm import tqdm

from torch import nn, svd
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score, precision_recall_curve

import wandb

wandb.init(project="test-project", entity="graphbois")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## HYPERPARAMETERS
# TODO implement an arg_parser module for the hyperparameters

# Default values
# embedding_size,dropout,lr,weight_decay, svdComponents, thirds=128,0.3,1e-3,5e-3, 100, False

# Current Values
embedding_size = 96
dropout = 0.3
lr = 1e-3
weight_decay = 5e-3
svdComponents = 100
thirds = False
epochs = 60

## IMPORTING THE DATASET
print("importing the dataset...")
# dataset=Twibot20(device=device,process=True,save=True)
# dataset = TwibotSmallTruncatedSVD(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset = TwibotSmallAugmentedTSVDHomogeneous(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset = HeteroTwibot(dataset)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()

## IMPORTING THE MODEL
print("setting up the model...")
# model = TweetAugmentedHAN(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents).to(device)
model = TweetAugmentedRGCN(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents).to(device)

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


wandb.watch(model)


# model=BotRGCN(embedding_dimension=embedding_size, des_size=svdComponents, tweet_size=svdComponents).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)

def train(epoch):
    model.train()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    loss_val = loss(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    wandb.log({"loss_train": loss_train, "acc_train": acc_train, "acc_val": acc_val, "loss_val": loss_val})

    if (epoch+1)  % 5 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

def test():
    model.eval()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output = output.max(1)[1].to('cpu').detach().numpy()
    label = labels.to('cpu').detach().numpy()
    f1 = f1_score(label[test_idx],output[test_idx])
    mcc = matthews_corrcoef(label[test_idx], output[test_idx])
    prec = precision_score(label[test_idx], output[test_idx])
    recall = recall_score(label[test_idx], output[test_idx])
    roc_auc = roc_auc_score(label[test_idx], output[test_idx])
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "f1_score= {:.4f}".format(f1.item()),
            "mcc= {:.4f}".format(mcc.item()),
            )

    wandb.log({"loss_test": loss_test, "acc_test": acc_test, "f1_score": f1, "mcc": mcc, "prec": prec, "recall": recall, "roc_auc": roc_auc})

    # Optional
    
    return acc_test,loss_test,f1

model.apply(init_weights)

print("beginning training...")

for epoch in tqdm(range(epochs), miniters=5):
    train(epoch)
    
test()