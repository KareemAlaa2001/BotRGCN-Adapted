from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD
import torch
from torch_geometric.data import HeteroData


class HeteroTwibot():
    def __init__(self, edgeHetero: TwibotSmallEdgeHetero):
        des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=edgeHetero.dataloader()

        self.data = HeteroData()

        self.data['user'].des = des_tensor
        self.data['user'].cat = category_prop
        self.data['user'].num = num_prop
        # self.data['user'].x = None
        self.data['user'].x = torch.cat((des_tensor, category_prop, num_prop), dim=1)
        self.data['user'].y = labels
        self.data['user'].train_idx = train_idx
        self.data['user'].val_idx = val_idx
        self.data['user'].test_idx = test_idx
        self.data['tweet'].x = tweets_tensor

        self.data['user', 'following', 'user'].edge_index = edge_index[:, (edge_type == 0).nonzero().squeeze()]
        self.data['user', 'followedBy', 'user'].edge_index = edge_index[:, (edge_type == 1).nonzero().squeeze()]
        self.data['tweet', 'mentions', 'user'].edge_index = edge_index[:, (edge_type == 2).nonzero().squeeze()]
        self.data['tweet', 'mentions', 'user'].edge_index[0] = self.data['tweet', 'mentions', 'user'].edge_index[0] - self.data['user'].x.shape[0]
        self.data['tweet', 'retweets', 'user'].edge_index = edge_index[:, (edge_type == 3).nonzero().squeeze()]
        self.data['tweet', 'retweets', 'user'].edge_index[0] = self.data['tweet', 'retweets', 'user'].edge_index[0] - self.data['user'].x.shape[0]
        self.data['user','writes', 'tweet'].edge_index = edge_index[:, (edge_type == 4).nonzero().squeeze()]
        self.data['user','writes', 'tweet'].edge_index[1] = self.data['user','writes', 'tweet'].edge_index[1] - self.data['user'].x.shape[0]

    # def dataloader():
    #     return

def initializeHeteroAugTwibot(edgeHetero: TwibotSmallEdgeHetero) -> HeteroData:
    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=edgeHetero.dataloader()
    
    data = HeteroData()

    data['user'].des = des_tensor
    data['user'].cat = category_prop
    data['user'].num = num_prop
    # self.data['user'].x = None
    data['user'].x = torch.cat((des_tensor, category_prop, num_prop), dim=1)
    data['user'].y = labels

    data['user'].train_idx = train_idx
    data['user'].train_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].train_mask[train_idx] = 1


    data['user'].val_idx = val_idx
    data['user'].val_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].val_mask[val_idx] = 1

    data['user'].test_idx = test_idx
    data['user'].test_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].test_mask[test_idx] = 1
    
    data['tweet'].x = tweets_tensor

    data['user', 'following', 'user'].edge_index = edge_index[:, (edge_type == 0).nonzero().squeeze()]
    data['user', 'followedBy', 'user'].edge_index = edge_index[:, (edge_type == 1).nonzero().squeeze()]
    data['tweet', 'mentions', 'user'].edge_index = edge_index[:, (edge_type == 2).nonzero().squeeze()]
    data['tweet', 'retweets', 'user'].edge_index = edge_index[:, (edge_type == 3).nonzero().squeeze()]
    data['user','writes', 'tweet'].edge_index = edge_index[:, (edge_type == 4).nonzero().squeeze()]

    data['tweet', 'mentions', 'user'].edge_index[0] = data['tweet', 'mentions', 'user'].edge_index[0] - data['user'].x.shape[0]
    data['tweet', 'retweets', 'user'].edge_index[0] = data['tweet', 'retweets', 'user'].edge_index[0] - data['user'].x.shape[0]
    data['user','writes', 'tweet'].edge_index[1] = data['user','writes', 'tweet'].edge_index[1] - data['user'].x.shape[0]

    return data

def initEdgeHeteroAugTwibot(edgeHetero: TwibotSmallEdgeHetero) -> HeteroData:
    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=edgeHetero.dataloader()
    
    tweets_tensor = torch.cat((tweets_tensor, torch.zeros(tweets_tensor.shape[0], (num_prop.shape[1] + category_prop.shape[1]))), dim=1)

    data = HeteroData()

    data['user'].des = torch.cat((des_tensor, tweets_tensor), dim=0)
    data['user'].cat = torch.cat((category_prop, torch.zeros(tweets_tensor.shape[0], category_prop.shape[1])), dim=0)
    data['user'].num = torch.cat((num_prop, torch.zeros(tweets_tensor.shape[0], num_prop.shape[1])), dim=0)

    # self.data['user'].x = None
    data['user'].x = torch.cat((des_tensor, category_prop, num_prop), dim=1)
    data['user'].y = torch.cat((labels, torch.zeros(tweets_tensor.shape[0], labels.shape[1])), dim=0)

    data['user'].train_idx = train_idx
    data['user'].train_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].train_mask[train_idx] = 1


    data['user'].val_idx = val_idx
    data['user'].val_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].val_mask[val_idx] = 1

    data['user'].test_idx = test_idx
    data['user'].test_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].test_mask[test_idx] = 1

    data['user', 'following', 'user'].edge_index = edge_index[:, (edge_type == 0).nonzero().squeeze()]
    data['user', 'followedBy', 'user'].edge_index = edge_index[:, (edge_type == 1).nonzero().squeeze()]
    data['user', 'mentions', 'user'].edge_index = edge_index[:, (edge_type == 2).nonzero().squeeze()]
    data['user', 'retweets', 'user'].edge_index = edge_index[:, (edge_type == 3).nonzero().squeeze()]
    data['user','writes', 'user'].edge_index = edge_index[:, (edge_type == 4).nonzero().squeeze()]

    return data

def initHomoAugTwibot(edgeHetero: TwibotSmallEdgeHetero) -> HeteroData:
    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=edgeHetero.dataloader()

    data = HeteroData()

    data['user'].des = torch.cat((des_tensor, tweets_tensor), dim=0)
    data['user'].cat = torch.cat((category_prop, torch.zeros(tweets_tensor.shape[0], category_prop.shape[1])), dim=0)
    data['user'].num = torch.cat((num_prop, torch.zeros(tweets_tensor.shape[0], num_prop.shape[1])), dim=0)

    # self.data['user'].x = None
    data['user'].x = torch.cat((des_tensor, category_prop, num_prop), dim=1)
    data['user'].y = torch.cat((labels, torch.zeros(tweets_tensor.shape[0], labels.shape[1])), dim=0)

    data['user'].train_idx = train_idx
    data['user'].train_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].train_mask[train_idx] = 1


    data['user'].val_idx = val_idx
    data['user'].val_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].val_mask[val_idx] = 1

    data['user'].test_idx = test_idx
    data['user'].test_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].test_mask[test_idx] = 1

    data['user', 'links', 'user'].edge_index = edge_index

    return data

def initHomoTwibotNonAug(nonAug: TwibotSmallTruncatedSVD) -> HeteroData:
    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=nonAug.dataloader()

    data = HeteroData()

    data['user'].des = des_tensor
    data['user'].cat = category_prop
    data['user'].num = num_prop
    data['user'].tweet = tweets_tensor

    # self.data['user'].x = None
    data['user'].x = torch.cat((des_tensor, category_prop, num_prop, tweets_tensor), dim=1)
    data['user'].y = labels

    data['user'].train_idx = train_idx
    data['user'].train_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].train_mask[train_idx] = 1


    data['user'].val_idx = val_idx
    data['user'].val_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].val_mask[val_idx] = 1

    data['user'].test_idx = test_idx
    data['user'].test_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].test_mask[test_idx] = 1

    data['user', 'links', 'user'].edge_index = edge_index

    return data

def initEdgeHeteroTwibotNonAug(nonAug: TwibotSmallTruncatedSVD) -> HeteroData:
    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=nonAug.dataloader()

    data = HeteroData()

    data['user'].des = des_tensor
    data['user'].cat = category_prop
    data['user'].num = num_prop
    data['user'].tweet = tweets_tensor

    # self.data['user'].x = None
    data['user'].x = torch.cat((des_tensor, category_prop, num_prop, tweets_tensor), dim=1)
    data['user'].y = labels

    data['user'].train_idx = train_idx
    data['user'].train_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].train_mask[train_idx] = 1


    data['user'].val_idx = val_idx
    data['user'].val_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].val_mask[val_idx] = 1

    data['user'].test_idx = test_idx
    data['user'].test_mask = torch.zeros(data['user'].x.shape[0], dtype=torch.bool)
    data['user'].test_mask[test_idx] = 1

    data['user', 'following', 'user'].edge_index = edge_index[:, (edge_type == 0).nonzero().squeeze()]
    data['user', 'followedBy', 'user'].edge_index = edge_index[:, (edge_type == 1).nonzero().squeeze()]
    
    return data

