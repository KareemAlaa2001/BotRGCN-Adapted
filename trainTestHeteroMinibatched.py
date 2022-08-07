from HeteroTwibot import HeteroTwibot, initializeHeteroAugTwibot
from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero

from torch_geometric.loader import DataLoader, NeighborLoader

from augmodels import TweetAugmentedHAN, TweetAugmentedRGCN, TweetAugmentedHAN2, TweetAugmentedHAN2ExtraLayer, TweetAugHetGCN, TweetAugHANConfigurable
from model import BotRGCN
from trainTestHetero import test

import torch
from tqdm import tqdm

from torch import nn, svd
from utils import accuracy,init_weights

from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, precision_recall_curve, confusion_matrix
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
        total_loss += loss_batch.detach() * batch_size
        total_acc += acc_batch.detach() * batch_size

    
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
        total_loss += loss_batch.detach() * batch_size
        total_acc += acc_batch.detach() * batch_size

    test_loss = total_loss / total_examples 
    test_acc = total_acc / total_examples

    return test_loss, test_acc



## TODO FIX THIS FUNCTION, METRIC APPLICATION BROKEN
# def test_minibatched_with_metrics(loader, model, loss, device, **metrics):
#     model.eval()
#     total_examples = total_loss = total_acc = 0

#     metric_totals = {metric: 0.0 for metric in metrics}

#     total_conf_matrix = torch.zeros(2, 2)

#     for data in loader:
#         data = data.to(device, 'edge_index')

#         batch_size = data['user'].batch_size
#         output = model(data)

#         loss_batch = loss(output[:batch_size], data['user'].y[:batch_size])
#         acc_batch = accuracy(output[:batch_size],  data['user'].y[:batch_size])

#         total_loss += loss_batch * batch_size
#         total_acc += acc_batch * batch_size

#         total_examples += batch_size

#         y_pred = output[:batch_size].max(1)[1].detach().numpy()
#         y_true = data['user'].y[:batch_size].detach().numpy()



#         for metric in metrics:
#             try:
#                 metric_totals[metric] += metrics[metric](y_pred, y_true) * batch_size
#             except ValueError:
#                 print("Got the valueerror from a batch not containing both classes, continuing..")
#                 continue
#         # f1_batch = f1_score(y_true, y_pred, average='macro')
#         # mcc_batch = matthews_corrcoef(y_true, y_pred)
#         # precision_batch = precision_score(y_true, y_pred, average='macro')
#         # recall_batch = recall_score(y_true, y_pred, average='macro')
#         # roc_auc_batch = roc_auc_score(y_true, y_pred, average='macro')

#         # total_f1 += f1_batch * batch_size
#         # total_mcc += mcc_batch * batch_size
#         # total_precision += precision_batch * batch_size
#         # total_recall += recall_batch * batch_size
#         # total_roc_auc += roc_auc_batch * batch_size

#         conf_matrix_batch = confusion_matrix(y_true, y_pred)
#         total_conf_matrix += conf_matrix_batch

#     test_loss = total_loss / total_examples
#     test_acc = total_acc / total_examples

#     results = {metric: metric_totals[metric] / total_examples for metric in metrics}
#     results['conf_matrix'] = total_conf_matrix
#     results['loss_test'] = test_loss
#     results['acc_test'] = test_acc

#     return results


def trainTestHeteroMinibatched(embedding_size = 128, dropout = 0.3, lr = 1e-3, weight_decay = 5e-3, svdComponents = 100, thirds = False, epochs = 100, extraLayer=True, numHanLayers = 2, neighboursPerNode = 50, batch_size = 256, testing_enabled = True, using_external_config = False):
    wandb.init(project="test-project", entity="graphbois")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    ## IMPORTING THE DATASET
    print("importing the dataset...")

    dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
    dataset = initializeHeteroAugTwibot(dataset).to(device, 'x', 'y')

    # min(torch.cuda.device_count(),4) if torch.cuda.device_count() > 0 else 1
    kwargs = {'num_workers': min(torch.cuda.device_count(),4) if torch.cuda.device_count() > 0 and torch.device.type == 'cuda' else 1, 'persistent_workers': True, 'batch_size': batch_size}
    kwargs_test = {'num_workers': min(torch.cuda.device_count(),4) if torch.cuda.device_count() > 0 and torch.device.type == 'cuda' else 1, 'persistent_workers': True, 'batch_size': ((len(dataset['user'].test_idx) // 2) + 2)}
    train_loader = NeighborLoader(dataset, num_neighbors=[neighboursPerNode] * numHanLayers,shuffle=True, input_nodes=('user',dataset['user'].train_mask), **kwargs)
    val_loader = NeighborLoader(dataset, num_neighbors=[neighboursPerNode] * numHanLayers,shuffle=True, input_nodes=('user',dataset['user'].val_mask), **kwargs)
    test_loader = NeighborLoader(dataset, num_neighbors=[neighboursPerNode] * numHanLayers,shuffle=True, input_nodes=('user',dataset['user'].test_mask), **kwargs_test)
    

    # model = TweetAugmentedHAN2ExtraLayer(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents, metadata=dataset.metadata()).to(device)
    model = TweetAugHANConfigurable(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents, metadata=dataset.metadata(), extraLayer=extraLayer,numHanLayers=numHanLayers).to(device)
    # model = TweetAugmentedHAN(embedding_dimension=embedding_size,des_size=svdComponents, twvdComponents, metadeet_size=sata=dataset.metadata()).to(device)

    if not using_external_config:
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
                'loss_val: {:.4f}'.format(val_loss.item()),
                'acc_val: {:.4f}'.format(val_acc.item()))

    metrics = {'f1_score': f1_score, 'mcc': matthews_corrcoef, 'prec': precision_score, 'recall': recall_score, 'roc_auc': roc_auc_score}
    
    
    # results = test_minibatched_with_metrics(test_loader, model, loss, device, **metrics)
    # wandb.log(results)
    
    if testing_enabled:

        acc_test,loss_test,f1, roc_auc = test(model, dataset, loss)

        return acc_test
    else:
        return val_acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## HYPERPARAMETERS
    # TODO implement an arg_parser module for the hyperparameters

    # Default values
    # embedding_size,dropout,lr,weight_decay, svdComponents, thirds=128,0.3,1e-3,5e-3, 100, False

    # Current Values
    config_defaults = dict(
        model_name="TweetAugHANConfigurable",
        dataset="HeteroTwibot",
        embedding_size = 204,
        dropout = 0.18380768518137663,
        lr = 0.004164987490510339,
        weight_decay = 0.0027187218127487783,
        svdComponents = 100,
        thirds = True,
        epochs = 60,
        extraLayer = True,
        numHANLayers = 4,
        neighboursPerNode = 382,
        batch_size = 256,
        testing_enabled = False,
    )

    wandb.init(project="test-project", entity="graphbois",  config=config_defaults)

    config = wandb.config

    trainTestHeteroMinibatched(config.embedding_size, config.dropout, config.lr, config.weight_decay, config.svdComponents, config.thirds, config.epochs, config.extraLayer, config.numHANLayers, config.neighboursPerNode, config.batch_size, config.testing_enabled, using_external_config=True)


"""
Hyperparameter Search Winner Winner Chicken Dinner
TODO run this one on server with more epochs and see if it improves
    {
    "lr": {
        "desc": null,
        "value": 0.004164987490510339
    },
    "_wandb": {
        "desc": null,
        "value": {
            "t": {
                "1": [
                    1,
                    5,
                    53,
                    55,
                    77
                ],
                "3": [
                    16,
                    24
                ],
                "4": "3.7.11",
                "5": "0.12.21",
                "8": [
                    5
                ]
            },
            "framework": "torch",
            "start_time": 1659295416,
            "cli_version": "0.12.21",
            "is_jupyter_run": false,
            "python_version": "3.7.11",
            "is_kaggle_kernel": false
        }
    },
    "epochs": {
        "desc": null,
        "value": 34
    },
    "thirds": {
        "desc": null,
        "value": false
    },
    "dataset": {
        "desc": null,
        "value": "HeteroTwibot"
    },
    "dropout": {
        "desc": null,
        "value": 0.18380768518137663
    },
    "batch_size": {
        "desc": null,
        "value": 256
    },
    "extraLayer": {
        "desc": null,
        "value": true
    },
    "model_name": {
        "desc": null,
        "value": "TweetAugHANConfigurable"
    },
    "numHANLayers": {
        "desc": null,
        "value": 3
    },
    "numHanLayers": {
        "desc": null,
        "value": 4
    },
    "weight_decay": {
        "desc": null,
        "value": 0.0027187218127487783
    },
    "svdComponents": {
        "desc": null,
        "value": 50
    },
    "embedding_size": {
        "desc": null,
        "value": 204
    },
    "testing_enabled": {
        "desc": null,
        "value": false
    },
    "neighboursPerNode": {
        "desc": null,
        "value": 382
    }
}
"""