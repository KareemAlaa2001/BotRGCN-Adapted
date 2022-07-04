import torch
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD
from TwibotSmallAugmentedTSVDHomogeneous import TwibotSmallAugmentedTSVDHomogeneous
from TwibotSmallEdgeHetero import TwibotSmallEdgeHetero
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size,dropout,lr,weight_decay,svdComponents=128,0.3,1e-3,5e-3,100

# dataset=Twibot20(device=device,process=True,save=True)
# dataset = TwibotSmallTruncatedSVD(device=device,process=True,save=True,dev=True)
# dataset = TwibotSmallAugmentedTSVDHomogeneous(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
dataset = TwibotSmallEdgeHetero(device=device,process=True,save=True,dev=False, svdComponents=svdComponents)
# dataset.test_tweet_mentions()
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()