import torch
from torch import nn
from torch_geometric.nn import RGCNConv,FastRGCNConv,GCNConv,GATConv
import torch.nn.functional as F

# TODO
"""
Following along the BotRGCN blueprint but making some changes:
Rather than having a uniform embedding for all 4 parts, 

we have a user embedding containing num_prop, cat_prop and descriptions
each part is represented by embedding dim/3. 
Alternatively, could experiment with weighing these different components differently
(e.g. 1/4,1/4, 1/2)

Then have a dense layer projecting the tweet and user embeddings to the same dimension, so that they can finally be concatenated to for the combined tensor for the GNN
This relies on the edge_index matrix using combined indices (and me keeping track of them in the first place)


TODO currently relying on an implementation where the tweets are projected to the embedding_dimension, which should thus be smaller than the tweet's svdComponents
TODO this model only really works for homogeneous nodes, need to implement one for fully heterogeneous graphs
"""
class TweetAugmentedRGCN(nn.Module):
    def __init__(self,des_size=100,tweet_size=100,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3, thirds=False, additional_tweet_features=False):
        super(TweetAugmentedRGCN, self).__init__()
        self.dropout = dropout

        ## TODO this is a stop-gap solution, but there should be a more rhobust way of experimenting with the embedding sizes
        if thirds:
            if embedding_dimension%3!=0:
                raise ValueError("embedding_dimension must be divisible by 3")

            self.num_prop_size = embedding_dimension // 3
            self.cat_prop_size = embedding_dimension // 3
            self.des_size = embedding_dimension // 3
        
        else:
            if embedding_dimension%4!=0:
                raise ValueError("embedding_dimension must be divisible by 4")
            self.num_prop_size = embedding_dimension // 4
            self.cat_prop_size = embedding_dimension // 4
            self.des_size = embedding_dimension // 2

        if additional_tweet_features:
            raise ValueError("additional_tweet_features not yet implemented")
        else:
            self.tweet_size = embedding_dimension


        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,self.des_size),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,self.num_prop_size),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,self.cat_prop_size),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)

    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,n,c),dim=1)
        
        # Homo graph solution: user features together form a (num_users, embedding_size) matrix which is concatenated to the tweet matrix of size (num_tweets, embedding_size)
        # print(tweet.dtype)
        # print(tweet.shape)
        t=self.linear_relu_tweet(tweet.float())
        x=torch.cat((x,t),dim=0)


        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x

class BotRGCN(nn.Module):
    def __init__(self,des_size=100,tweet_size=100,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class BotRGCN1(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN1, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        ''''
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        '''
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=d
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class BotRGCN2(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN2, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        '''
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=t
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class BotRGCN3(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN3, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        '''
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=n
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class BotRGCN4(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN4, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        '''
        c=self.linear_relu_cat_prop(cat_prop)
        x=c
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotRGCN12(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN12, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=torch.cat((d,t),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotRGCN34(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN34, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotGCN(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gcn1=GCNConv(embedding_dimension,embedding_dimension)
        self.gcn2=GCNConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gcn1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gcn2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotGAT(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotGAT, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gat1=GATConv(embedding_dimension,int(embedding_dimension/4),heads=4)
        self.gat2=GATConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gat1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gat2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
class BotRGCN_4layers(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN_4layers, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    

class BotRGCN_8layers(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN_8layers, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x