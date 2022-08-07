import torch
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD
from TwibotSmallAugmentedTSVDHomogeneous import TwibotSmallAugmentedTSVDHomogeneous
from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero
from HeteroTwibot import initializeHeteroAugTwibot, initEdgeHeteroTwibotNonAug
from torch_geometric.loader import DataLoader, NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size,dropout,lr,weight_decay,svdComponents=128,0.3,1e-3,5e-3,50
num_neighbors = 50
numHanLayers = 2
# dataset=Twibot20(device=device,process=True,save=True)
dataset = TwibotSmallTruncatedSVD(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset = TwibotSmallAugmentedTSVDHomogeneous(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset = initializeHeteroAugTwibot(dataset)
dataset = initEdgeHeteroTwibotNonAug(dataset)

print("num users", dataset.x_dict['user'].shape)

kwargs = {'num_workers': torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1, 'persistent_workers': True, 'batch_size': 1024}
kwargs_test = {'num_workers': min(torch.cuda.device_count(),4) if torch.cuda.device_count() > 0 else 1, 'persistent_workers': True, 'batch_size': ((len(dataset['user'].test_idx) // 2) + 2)}


train_loader = NeighborLoader(dataset, num_neighbors=[50] * numHanLayers,shuffle=False, input_nodes=('user',dataset['user'].train_mask), **kwargs)
val_loader = NeighborLoader(dataset, num_neighbors=[50] * numHanLayers,shuffle=False, input_nodes=('user',dataset['user'].val_mask), **kwargs)
test_loader = NeighborLoader(dataset, num_neighbors=[50] * numHanLayers,shuffle=False, input_nodes=('user',dataset['user'].test_mask), **kwargs_test)
for data in test_loader:
    print(type(data))
    print(data.x_dict['user'].shape)
    print(data.x_dict['tweet'].shape)
    print(data['user'].batch_size)

print(len(test_loader))

# print(loader)
# dataset.test_tweet_mentions()
# des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()