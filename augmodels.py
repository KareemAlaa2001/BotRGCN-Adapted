import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv,FastRGCNConv,GCNConv,GATConv, HANConv, Linear, HeteroLinear, to_hetero


class TweetAugHANConfigurable(nn.Module):
    def __init__(self,des_size=100,tweet_size=100,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3, thirds=False, augmented_data=True, metadata=None, extraLayer=True, numHanLayers=1):
        super(TweetAugHANConfigurable, self).__init__()
        self.dropout = dropout
        self.extraLayer = extraLayer
        self.augmented_data = augmented_data
        if not metadata:
            raise ValueError("Must provide metadata")

        if numHanLayers < 1:
            raise ValueError("Must have at least one HAN layer")

        self.numHanLayers = numHanLayers
        ## TODO this is a stop-gap solution, but there should be a more rhobust way of experimenting with the embedding sizes (let's just keep the stopgap lmao)
        if thirds:
            if not augmented_data:
                raise ValueError("Cannot use thirds with unaugmented data")
            if embedding_dimension%3!=0:
                raise ValueError("embedding_dimension must be divisible by 3")

            self.num_prop_size = embedding_dimension // 3
            self.cat_prop_size = embedding_dimension // 3
            self.des_size = embedding_dimension // 3
            self.tweet_size = embedding_dimension // 3
        
        else:
            if embedding_dimension%4!=0:
                raise ValueError("embedding_dimension must be divisible by 4")
            
            if augmented_data:
                self.num_prop_size = embedding_dimension // 4
                self.cat_prop_size = embedding_dimension // 4
                self.des_size = embedding_dimension // 2
                self.tweet_size = embedding_dimension // 2
            else:
                self.num_prop_size = embedding_dimension // 4
                self.cat_prop_size = embedding_dimension // 4
                self.des_size = embedding_dimension // 4
                self.tweet_size = embedding_dimension // 4

        # if additional_tweet_features:
        #     raise ValueError("additional_tweet_features not yet implemented")
        # else:
        #     self.tweet_size = embedding_dimension


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
            nn.Linear(tweet_size,self.tweet_size),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.han = HANConv(in_channels=-1, out_channels=embedding_dimension, metadata=metadata, dropout=self.dropout)
        # self.han=HANConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.heteroLinear_output1 = HeteroLinear(-1,embedding_dimension,num_types=2)
        self.relu_output1=nn.Sequential(
            nn.LeakyReLU()
        )
        self.heteroLinear_output2=HeteroLinear(embedding_dimension,2, num_types=2)


    def forward(self,data: HeteroData):
        d=self.linear_relu_des(data['user'].des)
        n=self.linear_relu_num_prop(data['user'].num)
        c=self.linear_relu_cat_prop(data['user'].cat)
        x=torch.cat((d,n,c),dim=1)

        
        if len(data.node_types) == 2:
            t=self.linear_relu_tweet(data['tweet'].x.float())
            out_dict = {'user':x,'tweet':t}

            ### NOTE making the extra layer an optional hyperparameter
            if self.extraLayer:
                out_dict['user'] = self.linear_relu_input(out_dict['user'])
                out_dict['tweet'] = self.linear_relu_input(out_dict['tweet'])


            out = self.han(out_dict, data.edge_index_dict)

            for i in range(self.numHanLayers-1):
                out = self.han(out, data.edge_index_dict)

            user_type_vec = torch.zeros(out['user'].shape[0])
            tweet_type_vec = torch.ones(out['tweet'].shape[0])
            type_vec = torch.cat((user_type_vec,tweet_type_vec),dim=0)
            out=self.heteroLinear_output1(torch.cat((out['user'], out['tweet']), dim=0), type_vec)
            
            out = self.relu_output1(out)
            out=self.heteroLinear_output2(out, type_vec)

        else:
            if len(data.node_types) != 1:
                raise ValueError("Must have 1 or 2 node types")

            # The problem here is with the minibatching not maintaing the relevant tweets for the users, 
            # need to depend on the join being completed earlier

            # so we know that if the dataset is augmented, the neighbourloader will give a batch of batch_size User nodes + a bunch of tweet/user nodes. 
            # Could depend on tweet nodes being last if we weren't minibatched, but that's not necessarilty the case. 
            # We do know that for tweets, des, num and cat are all zeros. Wait no, we should be applying out the same linear layer to all the nodes.
            # if augmented, there is no tweet info, it's all in the des embedding!!! So we can just proceed without the tweet info being extra-processed.
            if self.augmented_data == False:
                t = self.linear_relu_tweet(data['user'].tweet.float())
                x = torch.cat((x,t),dim=1)

            out_dict = {'user':x}

            if self.extraLayer:
                out_dict['user'] = self.linear_relu_input(out_dict['user'])
            
            out = self.han(out_dict, data.edge_index_dict)
            for i in range(self.numHanLayers-1):
                out = self.han(out, data.edge_index_dict)
            
            type_vec = torch.zeros(out['user'].shape[0])

            out=self.heteroLinear_output1(out['user'], type_vec)
            out = self.relu_output1(out)
            out=self.heteroLinear_output2(out, type_vec)

            # TODO adjust tweet embedding size after sorting out input
            # if data isnt augmented, then we use the same //4 for all component embedding sizes
            ## if it is, then we want to maintain the //2, //4, //4
            
        return out

class TweetAugHetGCN(nn.Module):
    def __init__(self,des_size=100,tweet_size=100,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3, thirds=False, additional_tweet_features=False, metadata=None):
        super(TweetAugHetGCN, self).__init__()
        self.dropout = dropout

        if not metadata:
            raise ValueError("Must provide metadata")

        ## TODO this is a stop-gap solution, but there should be a more robust way of experimenting with the embedding sizes
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
        
        self.hetGCN_conv2=to_hetero(nn.Sequential(
            GCNConv(embedding_dimension,embedding_dimension),
            nn.Dropout(self.dropout),
            GCNConv(embedding_dimension,embedding_dimension)
        ), metadata=metadata)

        # self.han = HANConv(in_channels=-1, out_channels=embedding_dimension, metadata=metadata, dropout=self.dropout)
        # self.han=HANConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.heteroLinear_output1 = HeteroLinear(-1,embedding_dimension,num_types=2)
        self.relu_output1=nn.Sequential(
            nn.LeakyReLU()
        )
        self.heteroLinear_output2=HeteroLinear(embedding_dimension,2, num_types=2)


    def forward(self,data: HeteroData):

        d=self.linear_relu_des(data['user'].des)
        n=self.linear_relu_num_prop(data['user'].num)
        c=self.linear_relu_cat_prop(data['user'].cat)
        x=torch.cat((d,n,c),dim=1)
        
        t=self.linear_relu_tweet(data['tweet'].x.float())
        # x=torch.cat((x,t),dim=0)
        # data['user'].x = x
        # data['tweet'].x = t
        out_dict = {'user':x,'tweet':t}
        ### NOTE removed the extra layer that was supposedly applied to ALL the nodes uniformly

        out = self.hetGCN_conv2(out_dict, data.edge_index_dict)

        user_type_vec = torch.zeros(out['user'].shape[0])
        tweet_type_vec = torch.ones(out['tweet'].shape[0])

        type_vec = torch.cat((user_type_vec,tweet_type_vec),dim=0)
        out=self.heteroLinear_output1(torch.cat((out['user'], out['tweet']), dim=0), type_vec)
        
        out = self.relu_output1(out)
        out=self.heteroLinear_output2(out, type_vec)
            
        return out



class TweetAugmentedHAN2ExtraLayer(nn.Module):
    def __init__(self,des_size=100,tweet_size=100,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3, thirds=False, additional_tweet_features=False, metadata=None):
        super(TweetAugmentedHAN2ExtraLayer, self).__init__()
        self.dropout = dropout

        if not metadata:
            raise ValueError("Must provide metadata")

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
        
        self.han = HANConv(in_channels=-1, out_channels=embedding_dimension, metadata=metadata, dropout=self.dropout)
        # self.han=HANConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.heteroLinear_output1 = HeteroLinear(-1,embedding_dimension,num_types=2)
        self.relu_output1=nn.Sequential(
            nn.LeakyReLU()
        )
        self.heteroLinear_output2=HeteroLinear(embedding_dimension,2, num_types=2)


    def forward(self,data: HeteroData):

        d=self.linear_relu_des(data['user'].des)
        n=self.linear_relu_num_prop(data['user'].num)
        c=self.linear_relu_cat_prop(data['user'].cat)
        x=torch.cat((d,n,c),dim=1)
        
        t=self.linear_relu_tweet(data['tweet'].x.float())
        # x=torch.cat((x,t),dim=0)
        # data['user'].x = x
        # data['tweet'].x = t
        out_dict = {'user':x,'tweet':t}
        ### NOTE experimenting with the extra layer that was applied to ALL the nodes uniformly
        out_dict['user'] = self.linear_relu_input(out_dict['user'])
        out_dict['tweet'] = self.linear_relu_input(out_dict['tweet'])
        ## NOTE experimenting with 2 HAN conv layers

        out = self.han(out_dict, data.edge_index_dict)
        out = self.han(out, data.edge_index_dict)

        user_type_vec = torch.zeros(out['user'].shape[0])
        tweet_type_vec = torch.ones(out['tweet'].shape[0])

        type_vec = torch.cat((user_type_vec,tweet_type_vec),dim=0)
        out=self.heteroLinear_output1(torch.cat((out['user'], out['tweet']), dim=0), type_vec)
        
        out = self.relu_output1(out)
        out=self.heteroLinear_output2(out, type_vec)
            
        return out

class TweetAugmentedHAN2(nn.Module):
    def __init__(self,des_size=100,tweet_size=100,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3, thirds=False, additional_tweet_features=False, metadata=None):
        super(TweetAugmentedHAN2, self).__init__()
        self.dropout = dropout

        if not metadata:
            raise ValueError("Must provide metadata")

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
        
        self.han = HANConv(in_channels=-1, out_channels=embedding_dimension, metadata=metadata, dropout=self.dropout)
        # self.han=HANConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.heteroLinear_output1 = HeteroLinear(-1,embedding_dimension,num_types=2)
        self.relu_output1=nn.Sequential(
            nn.LeakyReLU()
        )
        self.heteroLinear_output2=HeteroLinear(embedding_dimension,2, num_types=2)


    def forward(self,data: HeteroData):

        d=self.linear_relu_des(data['user'].des)
        n=self.linear_relu_num_prop(data['user'].num)
        c=self.linear_relu_cat_prop(data['user'].cat)
        x=torch.cat((d,n,c),dim=1)
        
        t=self.linear_relu_tweet(data['tweet'].x.float())
        # x=torch.cat((x,t),dim=0)
        # data['user'].x = x
        # data['tweet'].x = t
        out_dict = {'user':x,'tweet':t}
        ### NOTE removed the extra layer that was supposedly applied to ALL the nodes uniformly

        ## NOTE experimenting with 2 HAN conv layers

        out = self.han(out_dict, data.edge_index_dict)
        out = self.han(out, data.edge_index_dict)

        user_type_vec = torch.zeros(out['user'].shape[0])
        tweet_type_vec = torch.ones(out['tweet'].shape[0])

        type_vec = torch.cat((user_type_vec,tweet_type_vec),dim=0)
        out=self.heteroLinear_output1(torch.cat((out['user'], out['tweet']), dim=0), type_vec)
        
        out = self.relu_output1(out)
        out=self.heteroLinear_output2(out, type_vec)
            
        return out

class TweetAugmentedHAN(nn.Module):
    def __init__(self,des_size=100,tweet_size=100,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3, thirds=False, additional_tweet_features=False, metadata=None):
        super(TweetAugmentedHAN, self).__init__()
        self.dropout = dropout

        if not metadata:
            raise ValueError("Must provide metadata")

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
        
        # TODO check documentation for this TODO check if the 2 value for out_channels is valid if things break
        self.han = HANConv(in_channels=-1, out_channels=embedding_dimension, metadata=metadata, dropout=self.dropout)
        # self.han=HANConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.heteroLinear_output1 = HeteroLinear(-1,embedding_dimension,num_types=2)
        self.relu_output1=nn.Sequential(
            nn.LeakyReLU()
        )
        self.heteroLinear_output2=HeteroLinear(embedding_dimension,2, num_types=2)


    def forward(self,data: HeteroData):

        d=self.linear_relu_des(data['user'].des)
        n=self.linear_relu_num_prop(data['user'].num)
        c=self.linear_relu_cat_prop(data['user'].cat)

        x=torch.cat((d,n,c),dim=1)
        t=self.linear_relu_tweet(data['tweet'].x.float())
        out_dict = {'user':x,'tweet':t}
        ### NOTE removed the extra layer that was supposedly applied to ALL the nodes uniformly

        #### NOTE trying with one HAN layer

        out = self.han(out_dict, data.edge_index_dict)

        user_type_vec = torch.zeros(out['user'].shape[0])
        tweet_type_vec = torch.ones(out['tweet'].shape[0])
        type_vec = torch.cat((user_type_vec,tweet_type_vec),dim=0)
        out=self.heteroLinear_output1(torch.cat((out['user'], out['tweet']), dim=0), type_vec)
        
        out = self.relu_output1(out)
        out=self.heteroLinear_output2(out, type_vec)
            
        return out

"""
Following along the BotRGCN blueprint but making some changes:
Rather than having a uniform embedding for all 4 parts, 

we have a user embedding containing num_prop, cat_prop and descriptions
each part is represented by embedding dim/3. 
Alternatively, could experiment with weighing these different components differently
(e.g. 1/4,1/4, 1/2)

Then have a dense layer projecting the tweet and user embeddings to the same dimension, so that they can finally be concatenated to for the combined tensor for the GNN
This relies on the edge_index matrix using combined indices (and me keeping track of them in the first place)


NOTE currently relying on an implementation where the tweets are projected to the embedding_dimension
NOTE this model only really works for homogeneous nodes, using HAN fully heterogeneous graphs
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