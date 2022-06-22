import torch
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size,dropout,lr,weight_decay=128,0.3,1e-3,5e-3

# dataset=Twibot20(device=device,process=True,save=True)
dataset = TwibotSmallTruncatedSVD(device=device,process=True,save=True,dev=True)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()