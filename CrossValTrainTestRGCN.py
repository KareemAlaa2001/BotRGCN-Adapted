from model import BotRGCN
from augmodels import TweetAugmentedRGCN, TweetAugmentedHAN
# from Dataset import Twibot20
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD
from TwibotSmallAugmentedTSVDHomogeneous import TwibotSmallAugmentedTSVDHomogeneous
from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero
from HeteroTwibot import HeteroTwibot

import numpy as np

import torch
from tqdm import tqdm

from torch import nn, svd
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score, precision_recall_curve, confusion_matrix
import argparse

import wandb

# dataset=Twibot20(device=device,process=True,save=True)
# dataset = TwibotSmallTruncatedSVD(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset = TwibotSmallAugmentedTSVDHomogeneous(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset = HeteroTwibot(dataset)
# model=BotRGCN(embedding_dimension=embedding_size, des_size=svdComponents, tweet_size=svdComponents).to(device)


def train(epoch, model, optimizer, loss, des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx):
    model.train()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    return acc_train,loss_train
    

def test(model, loss, des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,\
    test_idx, **metrics):
    model.eval()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    output_probs = torch.softmax(output, dim=1).detach().cpu().numpy()
    loss_test = loss(output[test_idx], labels[test_idx]).detach()
    acc_test = accuracy(output[test_idx], labels[test_idx]).detach()
    output = output.max(1)[1].to('cpu').detach().numpy()
    label = labels.to('cpu').detach().numpy()
    
    results = {}

    for metric in metrics:
        if metric == 'roc_auc':
            results[metric] = metrics[metric](label[test_idx], output_probs[test_idx,1])
        results[metric] = metrics[metric](label[test_idx], output[test_idx])

    results['loss'] = loss_test.item()
    results['acc'] = acc_test.item()
    # if val_set:
    #     print("Validation set results:",
    #             "val_loss= {:.4f}".format(loss_test.item()),
    #             "val_accuracy= {:.4f}".format(acc_test.item()),
    #             "val_f1_score= {:.4f}".format(results['f1_score'].item()),
    #             "val_mcc= {:.4f}".format(results['mcc'].item()),
    #             "val_precision= {:.4f}".format(results['prec'].item()),
    #             "val_recall= {:.4f}".format(results['recall'].item()),
    #             "val_roc_auc= {:.4f}".format(results['roc_auc'].item()))
                
    # else:
    #     print("Test set results:",
    #             "test_loss= {:.4f}".format(loss_test.item()),
    #             "test_accuracy= {:.4f}".format(acc_test.item()),
    #             "f1_score= {:.4f}".format(results['f1_score'].item()),
    #             "mcc={:.4f}".format(results['mcc'].item()),
    #             "precision= {:.4f}".format(results['prec'].item()),
    #             "recall= {:.4f}".format(results['recall'].item()),
    #             "roc_auc= {:.4f}".format(results['roc_auc'].item()))

    # Optional
    
    return results


def crossValTrainTestBotRGCN(embedding_size = 128, dropout = 0.3, lr = 1e-3, weight_decay = 5e-3, svdComponents = 100, \
    thirds = False, epochs = 100, testing_enabled = True, using_external_config = False, augmentedDataset = True, datasetVariant = 1, crossValFolds = 5, crossValIteration=0, dev=False):
    
    device = torch.device('cpu')

    if datasetVariant not in {0,1}:
        raise ValueError("datasetVariant must be 1 or 0")
    
    numRelations = 2
    print("Importing the dataset...")
    if augmentedDataset:
        
        if datasetVariant == 0:
            dataset = TwibotSmallAugmentedTSVDHomogeneous(device=device,process=True,save=True,dev=dev, svdComponents=svdComponents)
            numRelations = 1
        else:
            dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
        
        model = TweetAugmentedRGCN(embedding_dimension=embedding_size, des_size=svdComponents, tweet_size=svdComponents, \
            dropout=dropout, thirds=thirds, numRelations=numRelations).to(device)
    else:
        if datasetVariant == 0:
            numRelations = 1

        dataset = TwibotSmallTruncatedSVD(device=device,process=True,save=True,dev=dev, svdComponents=svdComponents, edgeHetero=bool(datasetVariant))
        model = BotRGCN(embedding_dimension=embedding_size, des_size=svdComponents, tweet_size=svdComponents, dropout=dropout, numRelations=numRelations).to(device)

    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()

    assert crossValFolds > 1, "cross_val_folds must be greater than 1"
    assert crossValIteration < crossValFolds, "cross_val_iteration must be less than cross_val_folds"

    train_val_range = range(train_idx[0], val_idx[-1]+1)
    val_start_index = int((crossValIteration/crossValFolds) * len(train_val_range))
    val_end_index_exclusive = int(((crossValIteration+1)/crossValFolds) * len(train_val_range))

    val_idx = train_val_range[val_start_index:val_end_index_exclusive]
    train_idx = list(train_val_range[:val_start_index]) + list(train_val_range[val_end_index_exclusive:])


    ## IMPORTING THE MODEL
    print("setting up the model...")

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

    wandb.watch(model)

    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)

    model.apply(init_weights)

    print("beginning training...")

    metrics = {'f1_score': f1_score, 'mcc': matthews_corrcoef, 'prec': precision_score, \
        'recall': recall_score, 'roc_auc': roc_auc_score, 'conf_mat': confusion_matrix}
    for epoch in tqdm(range(epochs), miniters=5):
        acc_train,loss_train = train(epoch, model, optimizer, loss, des_tensor, tweets_tensor, \
            num_prop, category_prop, edge_index, edge_type, labels, train_idx)

        val_results = test(model, loss, des_tensor, tweets_tensor, \
            num_prop, category_prop, edge_index, edge_type, labels, \
                val_idx,**metrics)

        val_results_named = {k+"_val":v for k,v in val_results.items()}

        wandb.log({"acc_train": acc_train, "loss_train": loss_train, **val_results_named})
    results = val_results_named
    results['acc_train'] = acc_train
    results['loss_train'] = loss_train

    if testing_enabled:
        results = test(model, loss, des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,test_idx, val_set=False)
    
    return results        

def train_all_then_test_BotRGCN(embedding_size = 128, dropout = 0.3, lr = 1e-3, weight_decay = 5e-3, svdComponents = 100, \
    thirds = False, epochs = 100, testing_enabled = True, using_external_config = False, augmentedDataset = True, datasetVariant = 1, crossValFolds = 5, crossValIteration=0, dev=False):
    
    device = torch.device('cpu')

    if datasetVariant not in {0,1}:
        raise ValueError("datasetVariant must be 1 or 0")
    
    numRelations = 2
    print("Importing the dataset...")
    if augmentedDataset:
        
        if datasetVariant == 0:
            dataset = TwibotSmallAugmentedTSVDHomogeneous(device=device,process=True,save=True,dev=dev, svdComponents=svdComponents)
            numRelations = 1
        else:
            dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
        
        model = TweetAugmentedRGCN(embedding_dimension=embedding_size, des_size=svdComponents, tweet_size=svdComponents, \
            dropout=dropout, thirds=thirds, numRelations=numRelations).to(device)
    else:
        if datasetVariant == 0:
            numRelations = 1

        dataset = TwibotSmallTruncatedSVD(device=device,process=True,save=True,dev=dev, svdComponents=svdComponents, edgeHetero=bool(datasetVariant))
        model = BotRGCN(embedding_dimension=embedding_size, des_size=svdComponents, tweet_size=svdComponents, dropout=dropout, numRelations=numRelations).to(device)

    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()

    assert crossValFolds > 1, "cross_val_folds must be greater than 1"
    assert crossValIteration < crossValFolds, "cross_val_iteration must be less than cross_val_folds"

    train_val_idx = list(train_idx) + list(val_idx)

    ## IMPORTING THE MODEL
    print("setting up the model...")

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

    wandb.watch(model)

    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)

    model.apply(init_weights)

    print("beginning training...")

    metrics = {'f1_score': f1_score, 'mcc': matthews_corrcoef, 'prec': precision_score, \
        'recall': recall_score, 'roc_auc': roc_auc_score, 'conf_mat': confusion_matrix}

    for epoch in tqdm(range(epochs), miniters=5):
        acc_train,loss_train = train(epoch, model, optimizer, loss, des_tensor, tweets_tensor, \
            num_prop, category_prop, edge_index, edge_type, labels, train_val_idx)

        wandb.log({"acc_train": acc_train, "loss_train": loss_train})
    
    
    results = test(model, loss, des_tensor, tweets_tensor, \
            num_prop, category_prop, edge_index, edge_type, labels, \
                test_idx,**metrics)

    results_named = {k+"_test":v for k,v in results.items()}

    results_named['acc_train'] = acc_train.item()
    results_named['loss_train'] = loss_train.item()

    return results_named
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_variant', type=int, default=1, help='1 for edge heterogeneous, 0 for edge homogeneous')
    parser.add_argument('--augmented_dataset', type=bool, default=True, help='True for augmented dataset, False for non-augmented dataset')
    parser.add_argument('--test_not_val', type=bool, default=False, help='True for testing with val in train, false for running cross-val')
    args = parser.parse_args()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    ## HYPERPARAMETERS
    # TODO implement an arg_parser module for the hyperparameters

    # Default values
    # embedding_size,dropout,lr,weight_decay, svdComponents, thirds=128,0.3,1e-3,5e-3, 100, False

    # # OLD Values from main.py
    # embedding_size = 96
    # dropout = 0.3
    # lr = 1e-3
    # weight_decay = 5e-3
    # svdComponents = 100
    # thirds = False
    # epochs = 60

    # Current Values
    config_defaults = dict(
        model_name="BotRGCN",
        embedding_size = 96,
        dropout = 0.3,
        lr = 0.001,
        weight_decay = 0.005,
        svdComponents = 100,
        thirds = False,
        epochs = 60,
        # neighboursPerNode = 10,
        # batch_size=1,
        testing_enabled = args.test_not_val,
        crossValFolds = 5,
        augmentedDataset = args.augmented_dataset,
        datasetVariant = args.dataset_variant,
        dev = False,
        numRepeats = 10,
        numRepeatsPerFold = 3
    )


    wandb.init(project="test-project", entity="graphbois",  config=config_defaults)

    config = wandb.config

    
    aggregate_results = {}

    if config.testing_enabled:
        numRepeats = config.numRepeats

        for i in range(numRepeats):
            print("Starting repeat {}".format(i))
            results = train_all_then_test_BotRGCN(config.embedding_size, config.dropout, config.lr, \
                    config.weight_decay, config.svdComponents, config.thirds, config.epochs, config.testing_enabled, \
                            using_external_config=True, augmentedDataset=config.augmentedDataset, datasetVariant=config.datasetVariant, \
                                crossValFolds=config.crossValFolds, crossValIteration=i, dev=config.dev)
            wandb.log(results)
            for key in results:
                if key != 'conf_matrix_test':
                    aggregate_results[key] = aggregate_results.get(key, []) + [results[key]]
                else:
                    aggregate_results[key] = aggregate_results.get(key, []) + [results[key].numpy()]
    else:
        numRepeats = config.numRepeatsPerFold

        for i in range(config.crossValFolds):
            for j in range(numRepeats):
                val_results = crossValTrainTestBotRGCN(config.embedding_size, config.dropout, config.lr, \
                    config.weight_decay, config.svdComponents, config.thirds, config.epochs, config.testing_enabled, \
                            using_external_config=True, augmentedDataset=config.augmentedDataset, datasetVariant=config.datasetVariant, \
                                crossValFolds=config.crossValFolds, crossValIteration=i, dev=config.dev)
                
                for key in val_results:
                    if key not in aggregate_results:
                        aggregate_results[key] = []
                    
                    if key != 'conf_matrix_val':
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