from HeteroTwibot import HeteroTwibot, initializeHeteroAugTwibot, initHomoAugTwibot, initEdgeHeteroAugTwibot, initEdgeHeteroTwibotNonAug, initHomoTwibotNonAug
from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD
from trainTestHeteroMinibatched import train_minibatched

from torch_geometric.loader import DataLoader, NeighborLoader

from augmodels import TweetAugmentedHAN, TweetAugmentedRGCN, TweetAugmentedHAN2, TweetAugmentedHAN2ExtraLayer, TweetAugHetGCN, TweetAugHANConfigurable
from model import BotRGCN
from trainTestHetero import test

import torch
from tqdm import tqdm
import numpy as np 

from torch import nn, svd
from utils import accuracy,init_weights

from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, precision_recall_curve, confusion_matrix
import wandb


def test_minibatched_with_metrics(loader, model, loss, device, **metrics):
    model.eval()
    total_examples = total_loss = total_acc = 0

    metric_totals = {metric: 0.0 for metric in metrics}

    total_conf_matrix = torch.zeros((2,2))

    for data in loader:
        data = data.to(device, 'edge_index')

        batch_size = data['user'].batch_size
        output = model(data)

        loss_batch = loss(output[:batch_size], data['user'].y[:batch_size])
        acc_batch = accuracy(output[:batch_size],  data['user'].y[:batch_size])

        total_examples += batch_size
        total_loss += loss_batch.detach() * batch_size
        total_acc += acc_batch.detach() * batch_size

        y_pred = output[:batch_size].max(1)[1].detach().numpy()
        y_true = data['user'].y[:batch_size].detach().numpy()

        for metric in metrics:
            try:
                metric_totals[metric] += metrics[metric](y_pred, y_true) * batch_size
            except ValueError:
                print("Got the valueerror from a batch not containing both classes for metric",metric, "continuing..")
                continue
        
        conf_matrix_batch = confusion_matrix(y_true, y_pred)
        total_conf_matrix += conf_matrix_batch

    test_loss = total_loss / total_examples 
    test_acc = total_acc / total_examples
    
    results = {metric: metric_totals[metric] / total_examples for metric in metrics}
    results['conf_matrix'] = total_conf_matrix
    results['loss'] = test_loss
    results['acc'] = test_acc

    return results 


def trainValModelForCrossVal(embedding_size = 128, dropout = 0.3, lr = 1e-3, weight_decay = 5e-3, svdComponents = 100, \
    thirds = False, epochs = 100, extraLayer=True, numHanLayers = 2, neighboursPerNode = 50, batch_size = 256, testing_enabled = True, \
        using_external_config = False, augmentedDataset = True, datasetVariant = 2, crossValFolds = 5, crossValIteration=0, dev=False):
    wandb.init(project="test-project", entity="graphbois")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    ## IMPORTING THE DATASET
    print("importing the dataset...")


    if augmentedDataset:
        dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=dev, svdComponents=svdComponents)

        if datasetVariant == 0:
            dataset = initHomoAugTwibot(dataset, cross_val_enabled=True, cross_val_folds=crossValFolds, cross_val_iteration=crossValIteration).to(device, 'x', 'y')
            datasetName = "HomoAugTwibot"
        elif datasetVariant == 1:
            dataset = initEdgeHeteroAugTwibot(dataset, cross_val_enabled=True, cross_val_folds=crossValFolds, cross_val_iteration=crossValIteration).to(device, 'x', 'y')
            datasetName = "EdgeHeteroAugTwibot"
        elif datasetVariant == 2:
            dataset = initializeHeteroAugTwibot(dataset, cross_val_enabled=True, cross_val_folds=crossValFolds, cross_val_iteration=crossValIteration).to(device, 'x', 'y')
            datasetName = "HeteroAugTwibot"
        else:
            raise ValueError("datasetVariant must be 0,1 or 2")
    else:
        dataset = TwibotSmallTruncatedSVD(device=device,process=True,save=True,dev=dev, svdComponents=svdComponents)
        if datasetVariant == 0:
            dataset = initHomoTwibotNonAug(dataset, cross_val_enabled=True, cross_val_folds=crossValFolds, cross_val_iteration=crossValIteration).to(device, 'x', 'y')
            datasetName = "HomoTwibotNonAug"
        elif datasetVariant == 1:
            dataset = initEdgeHeteroTwibotNonAug(dataset, cross_val_enabled=True, cross_val_folds=crossValFolds, cross_val_iteration=crossValIteration).to(device, 'x', 'y')
            datasetName = "EdgeHeteroTwibotNonAug"
        else:
            raise ValueError("datasetVariant must be 0 or 1 for non augmented dataset")

    # min(torch.cuda.device_count(),4) if torch.cuda.device_count() > 0 else 1
    kwargs = {'num_workers': min(torch.cuda.device_count(),4) if torch.cuda.device_count() > 0 and torch.device.type == 'cuda' else 1, 'persistent_workers': True, 'batch_size': batch_size}
    kwargs_test = {'num_workers': min(torch.cuda.device_count(),4) if torch.cuda.device_count() > 0 and torch.device.type == 'cuda' else 1, 'persistent_workers': True, 'batch_size': ((len(dataset['user'].test_idx) // 2) + 2)}

    print("Getting loaders")
    train_loader = NeighborLoader(dataset, num_neighbors=[neighboursPerNode] * numHanLayers,shuffle=True, input_nodes=('user',dataset['user'].train_mask), **kwargs)
    print("Got train loader")
    val_loader = NeighborLoader(dataset, num_neighbors=[neighboursPerNode] * numHanLayers,shuffle=True, input_nodes=('user',dataset['user'].val_mask), **kwargs)
    print("Got val loader")
    test_loader = NeighborLoader(dataset, num_neighbors=[neighboursPerNode] * numHanLayers,shuffle=True, input_nodes=('user',dataset['user'].test_mask), **kwargs_test)
    print("Got test loader")
    
    print("Getting model")
    # model = TweetAugmentedHAN2ExtraLayer(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents, metadata=dataset.metadata()).to(device)
    model = TweetAugHANConfigurable(embedding_dimension=embedding_size,des_size=svdComponents, tweet_size=svdComponents, metadata=dataset.metadata(), extraLayer=extraLayer,numHanLayers=numHanLayers).to(device)
    # model = TweetAugmentedHAN(embedding_dimension=embedding_size,des_size=svdComponents, twvdComponents, metadeet_size=sata=dataset.metadata()).to(device)

    if not using_external_config:
        wandb.config.update({
        "model_name": model.__class__.__name__,
        "dataset": datasetName,
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

    metrics = {'f1_score': f1_score, 'mcc': matthews_corrcoef, 'prec': precision_score, \
        'recall': recall_score, 'roc_auc': roc_auc_score}

    for epoch in tqdm(range(epochs), miniters=5): 
        train_loss, train_acc = train_minibatched(epoch,model, train_loader, loss, optimizer, device)
        val_results = test_minibatched_with_metrics(val_loader, model, loss, device,**metrics)

        wandb.log({"loss_train": train_loss, "acc_train": train_acc, **val_results})

        if (epoch+1)  % 5 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(train_loss.item()),
                'acc_train: {:.4f}'.format(train_acc.item()),
                'loss_val: {:.4f}'.format(val_results['loss'].item()),
                'acc_val: {:.4f}'.format(val_results['acc'].item()))

    
    
    # results = test_minibatched_with_metrics(test_loader, model, loss, device, **metrics)
    # wandb.log(results)
    
    if testing_enabled:
        results = test_minibatched_with_metrics(test_loader, model, loss, device, **metrics)

        return results
    else:
        return val_results

if __name__ == '__main__':
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
        epochs = 1,
        extraLayer = True,
        numHANLayers = 4,
        neighboursPerNode = 382,
        batch_size = 256,
        # neighboursPerNode = 10,
        # batch_size=1,
        testing_enabled = False,
        crossValFolds = 3
    )

    wandb.init(project="test-project", entity="graphbois",  config=config_defaults)

    config = wandb.config

    aggregate_results = {}

    for i in range(config.crossValFolds):
        val_results = trainValModelForCrossVal(config.embedding_size, config.dropout, config.lr, \
            config.weight_decay, config.svdComponents, config.thirds, config.epochs, config.extraLayer, \
                config.numHANLayers, config.neighboursPerNode, config.batch_size, config.testing_enabled, \
                    using_external_config=True, augmentedDataset=True, datasetVariant=2, crossValFolds=5, \
                        crossValIteration=i, dev=False)
        
        for key in val_results:
            if key not in aggregate_results:
                aggregate_results[key] = []
            
            if key != 'conf_matrix':
                aggregate_results[key].append(val_results[key])
            else:
                aggregate_results[key].append(val_results[key].numpy())

    mean_results = {}
    result_stdev = {}
    print(aggregate_results)
    for key in aggregate_results:
        mean_results["mean_" + key] = np.array(aggregate_results[key]).mean(axis=0)
        result_stdev["stdev_" + key] = np.array(aggregate_results[key]).std(axis=0)

    wandb.log(mean_results)
    wandb.log(result_stdev)


        