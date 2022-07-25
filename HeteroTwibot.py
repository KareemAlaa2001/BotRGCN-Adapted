from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero
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

def initializeHeteroTwibot(edgeHetero: TwibotSmallEdgeHetero) -> HeteroData:
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