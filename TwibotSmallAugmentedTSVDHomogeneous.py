import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import os
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD
import re


"""
Constructing the dataset using separate tensors for users and tweets, which are handled by the model.

However, the construction of the graph has to be mindful of the combined indices from concatenating tweets to users further down the line for the formation of the graph.
For example, if we have 10k users and 200k tweets, want to be sure that an edge from user 2 to tweet 2 is shown as (2->10,002)
"""
class TwibotSmallAugmentedTSVDHomogeneous(TwibotSmallTruncatedSVD):

    def __init__(self, root='./Data/TwibotSmallAugmentedTSVDHomogeneous/',device=torch.device('cpu'), process=True, save=True, dev=False, svdComponents=100):
        self.root = root[:-1]+'Dev/' if dev else root

        if svdComponents != 100:
            self.root = self.root+'svd'+str(svdComponents)+'/'
            try: 
                os.mkdir(self.root) 
            except OSError as error: 
                # do nothing if the folder already exists
                pass
            
        self.device = device
        self.device_value = -1 if self.device.type =='cpu' else 0
        self.save=save
        self.svdComponents = svdComponents
        self.process=process
        self.dev = dev

        if process:
            
            if not dev:
                print('Loading train.json')
                df_train=pd.read_json('./Twibot-20/train.json')
                print('Loading test.json')
                df_test=pd.read_json('./Twibot-20/test.json')
                print('Small dataset version, not loading support.json')

                df_train=df_train.iloc[:,[0,1,2,3,5]]
                df_test=df_test.iloc[:,[0,1,2,3,5]]
                

            
            # df_support=pd.read_json('./Twibot-20/support.json')
            print('Loading dev.json')
            df_dev=pd.read_json('./Twibot-20/dev.json')
            print('Finished')
            df_dev=df_dev.iloc[:,[0,1,2,3,5]]

            if not dev:
                self.df_users = pd.concat([df_train,df_dev,df_test],ignore_index=True)

                # self.df_data_labeled=pd.concat([df_train,df_dev,df_test],ignore_index=True)
                # self.df_data=pd.concat([df_train,df_dev,df_test],ignore_index=True)
            else:
                self.df_users = df_dev
                # self.df_data_labeled=df_dev
                # self.df_data=df_dev
            
            self.df_tweet = None

    def load_labels(self):
        print('Loading labels...',end='   ')
        path=self.root+'label.pt'
        if not os.path.exists(path):
            labels=torch.LongTensor(self.df_users['label']).to(self.device)
            if self.save:
                torch.save(labels,self.root+'label.pt')
        else:
            labels=torch.load(self.root+"label.pt").to(self.device)
        print('Finished')
        
        return labels

    def tweets_embedding(self):
        print('Running tweet embedding')
        path=self.root+"tweets_tensor.pt"
        if not os.path.exists(path):
            if not self.df_tweet:
                self.df_tweet = self.extract_df_tweet()

            tweets = self.df_tweet['Body'].values
            vectorizer = TfidfVectorizer()
            tf_idf_matrix = vectorizer.fit_transform(tweets)
            csr_tfidf_matrix = csr_matrix(tf_idf_matrix)
            print("tf-idf matrix dimensions:", tf_idf_matrix.shape)
            print('fitting truncated SVD')
            svd = TruncatedSVD(n_components=self.svdComponents, n_iter=5, random_state=42)

            trunc_svd_matrix = svd.fit_transform(csr_tfidf_matrix)
            print("truncated SVD matrix dimensions:", trunc_svd_matrix.shape)

            tweets_tensor = torch.from_numpy(trunc_svd_matrix).to(self.device)
            if self.save:
                print('Saving tweets_tensor')
                torch.save(tweets_tensor,self.root+"tweets_tensor.pt")
        else:
            tweets_tensor=torch.load(self.root+"tweets_tensor.pt").to(self.device)
        print('Finished')

        return tweets_tensor

    def description_embedding(self):
        print("Loading user description embeddings")
        path = self.root+"des_tensor.pt"
        if not os.path.exists(path):
            descriptions = self.df_users['profile'].apply(lambda profile: profile['description'] if profile is not None and profile['description'] is not None else "None").values
            vectorizer = TfidfVectorizer()
            tf_idf_matrix = vectorizer.fit_transform(descriptions)
            csr_tfidf_matrix = csr_matrix(tf_idf_matrix)
            print("tf-idf matrix dimensions:", tf_idf_matrix.shape)
            print('fitting truncated SVD')
            svd = TruncatedSVD(n_components=self.svdComponents, n_iter=5, random_state=42) # n_iter=5 is the default, n_components=2 is the default
            # svd.fit(csr_tfidf_matrix)

            trunc_svd_matrix = svd.fit_transform(csr_tfidf_matrix)

            des_tensor=torch.Tensor(trunc_svd_matrix).to(self.device)
            if self.save:
                torch.save(des_tensor,self.root+'des_tensor.pt')
    
        else:
            des_tensor=torch.load(self.root+"des_tensor.pt").to(self.device)
        print('Finished')

        return des_tensor

    def num_prop_preprocess(self):
        print('Processing feature3...',end='   ')
        path0=self.root+'num_prop.pt'
        if not os.path.exists(path0):
            path=self.root
            if not os.path.exists(path+"followers_count.pt"):

                numerical_feature_names = ['followers_count', 'friends_count','favourites_count','statuses_count']
                followers_count, friends_count, favourites_count, statuses_count = [self.extractNumericalFeatureFromDf(feature_name, self.df_users).to(self.device) for feature_name in numerical_feature_names]
                print('typeof followers_count:',type(followers_count))
                print('followers_count shape:', followers_count.shape)
                if self.save:
                    for feature_name in numerical_feature_names:
                        torch.save(feature_name,path+feature_name+'.pt') 
                    
                ## TODO handle this separately from the other classes being cleaned up
                screen_name_length=[]
                for i in range (self.df_users.shape[0]):
                    if self.df_users['profile'][i] is None or self.df_users['profile'][i]['screen_name'] is None:
                        screen_name_length.append(0)
                    else:
                        screen_name_length.append(len(self.df_users['profile'][i]['screen_name']))
                screen_name_length=torch.tensor(np.array(screen_name_length,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(screen_name_length,path+'screen_name_length.pt')
                
                ## TODO handle this separately from the other classes being cleaned up
                active_days=[]
                date0=dt.strptime('Fri Jul 1 00:00:00 +0000 2022 ','%a %b %d %X %z %Y ')
                for i in range (self.df_users.shape[0]):
                    if self.df_users['profile'][i] is None or self.df_users['profile'][i]['created_at'] is None:
                        active_days.append(0)
                    else:
                        date=dt.strptime(self.df_users['profile'][i]['created_at'],'%a %b %d %X %z %Y ')
                        active_days.append((date0-date).days)
                active_days=torch.tensor(np.array(active_days,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(active_days,path+'active_days.pt')
                
                
            else:
                print('loading numerical properties from pt files')
                active_days=torch.load(path+"active_days.pt")
                screen_name_length=torch.load(path+"screen_name_length.pt")
                favourites_count=torch.load(path+"favourites_count.pt")
                followers_count=torch.load(path+"followers_count.pt")
                friends_count=torch.load(path+"friends_count.pt")
                statuses_count=torch.load(path+"statuses_count.pt")

            print('typeof followers_count:',type(followers_count))
            print(followers_count)
            print('followers_count shape:', followers_count.shape)
            
            
            active_days=pd.Series(active_days.to('cpu').detach().numpy())
            active_days=(active_days-active_days.mean())/active_days.std()
            active_days=torch.tensor(np.array(active_days))

            screen_name_length=pd.Series(screen_name_length.to('cpu').detach().numpy())
            screen_name_length_days=(screen_name_length-screen_name_length.mean())/screen_name_length.std()
            screen_name_length_days=torch.tensor(np.array(screen_name_length_days))

            favourites_count=pd.Series(favourites_count.to('cpu').detach().numpy())
            favourites_count=(favourites_count-favourites_count.mean())/favourites_count.std()
            favourites_count=torch.tensor(np.array(favourites_count))

            followers_count=pd.Series(followers_count.to('cpu').detach().numpy())
            followers_count=(followers_count-followers_count.mean())/followers_count.std()
            followers_count=torch.tensor(np.array(followers_count))

            friends_count=pd.Series(friends_count.to('cpu').detach().numpy())
            friends_count=(friends_count-friends_count.mean())/friends_count.std()
            friends_count=torch.tensor(np.array(friends_count))

            statuses_count=pd.Series(statuses_count.to('cpu').detach().numpy())
            statuses_count=(statuses_count-statuses_count.mean())/statuses_count.std()
            statuses_count=torch.tensor(np.array(statuses_count))

            num_prop=torch.cat((followers_count.reshape([self.df_users.shape[0],1]),friends_count.reshape([self.df_users.shape[0],1]),favourites_count.reshape([self.df_users.shape[0],1]),statuses_count.reshape([self.df_users.shape[0],1]),screen_name_length_days.reshape([self.df_users.shape[0],1]),active_days.reshape([self.df_users.shape[0],1])),1).to(self.device)

            if self.save:
                torch.save(num_prop,self.root + "num_prop.pt")
            
        else:
            num_prop=torch.load(self.root+"num_prop.pt").to(self.device)
        print('Finished')
        return num_prop

    def cat_prop_preprocess(self):
        print('Processing feature4...',end='   ')
        path=self.root+'category_properties.pt'
        if not os.path.exists(path):
            category_properties=[]
            properties=['protected','geo_enabled','verified','contributors_enabled','is_translator','is_translation_enabled','profile_background_tile','profile_use_background_image','has_extended_profile','default_profile','default_profile_image']
            for i in range (self.df_users.shape[0]):
                prop=[]
                if self.df_users['profile'][i] is None:
                    prop = [0] * len(properties)
                else:
                    for each in properties:
                        if self.df_users['profile'][i][each] is None:
                            prop.append(0)
                        else:
                            if self.df_users['profile'][i][each] == "True ":
                                prop.append(1)
                            else:
                                prop.append(0)

                prop=np.array(prop)
                category_properties.append(prop)
            category_properties=torch.tensor(np.array(category_properties,dtype=np.float32)).to(self.device)
            if self.save:
                torch.save(category_properties,self.root+'category_properties.pt')
        else:
            category_properties=torch.load(self.root+"category_properties.pt").to(self.device)
        print('Finished')
        return category_properties
    
    def Build_Graph(self):
        print('Building graph',end='   ')
        path=self.root+'edge_index.pt'
        if not os.path.exists(path):
            if not self.df_tweet:
                self.df_tweet = self.extract_df_tweet()

            id2index_dict={id:index for index,id in enumerate(self.df_users['ID'])}
            id2index_dict.update({id:index for index,id in enumerate(self.df_tweet['ID'], start=len(self.df_users['ID']))})

            df_user_profiles = self.df_users[self.df_users.profile.notnull()].apply(lambda x: pd.Series(x.profile), axis=1)
            uname2id_dict = {x['screen_name'].strip(): x['id'].strip() for index, x in df_user_profiles.iloc[:,[3,0]].transpose().to_dict().items()}

            edge_index=[]
            edge_type=[]

            # building the edges from the user relationships 
            for i,relation in enumerate(self.df_users['neighbor']):
                if relation is not None:
                    for each_id in relation['following']:
                        try:
                            target_index=id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i,target_index])
                        edge_type.append(0)
                    for each_id in relation['follower']:
                        try:
                            target_index=id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i,target_index])
                        # edge_type.append(1)
                        edge_type.append(0)
                else:
                    continue

            # building the edges from the tweet relationships

            for mentions, retweeted, tweeterid, tweetIndex in zip(self.df_tweet['mentions'], self.df_tweet['retweeted'], self.df_tweet['tweeterId'], self.df_tweet['tweetIndex']):
                for mention in mentions:
                    if uname2id_dict.get(mention) is not None:
                        if id2index_dict.get(int(uname2id_dict[mention])) is not None:
                            edge_index.append([int(tweetIndex),id2index_dict[int(uname2id_dict[mention])]])
                            # edge_type.append(2)
                            edge_type.append(0)

                for rt in retweeted:
                    if uname2id_dict.get(rt) is not None:
                        if id2index_dict.get(int(uname2id_dict[rt])) is not None:
                            target_index= id2index_dict[int(uname2id_dict[rt])]
                            edge_index.append([int(tweetIndex),target_index])
                            # edge_type.append(3)
                            edge_type.append(0)
        
                if id2index_dict.get(int(tweeterid)) is not None:
                    target_index=id2index_dict[int(tweeterid)]
                    edge_index.append([target_index,int(tweetIndex)])
                    # edge_type.append(4)
                    edge_type.append(0)
               
            edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(self.device)
            edge_type=torch.tensor(edge_type,dtype=torch.long).to(self.device)
            if self.save:
                torch.save(edge_index,self.root+"edge_index.pt")
                torch.save(edge_type,self.root+"edge_type.pt")
        else:
            edge_index=torch.load(self.root+"edge_index.pt").to(self.device)
            edge_type=torch.load(self.root+"edge_type.pt").to(self.device)
            print('Finished')
        return edge_index,edge_type
    
    def test_tweet_mentions(self):
        self.df_tweet['tweetIndex'] = self.df_tweet.index
        id2index_dict={id:index for index,id in enumerate(self.df_users['ID'])}
        id2index_dict.update({id:index for index,id in enumerate(self.df_tweet['ID'], start=len(self.df_users['ID']))})

        df_user_profiles = self.df_users[self.df_users.profile.notnull()].apply(lambda x: pd.Series(x.profile), axis=1)
        uname2id_dict = {x['screen_name'].strip(): x['id'].strip() for index, x in df_user_profiles.iloc[:,[3,0]].transpose().to_dict().items()}

        edge_index=[]
        edge_type=[]
        
        print(self.df_tweet.columns)
        print(self.df_tweet.shape)
        

    def dataloader(self):
        labels=self.load_labels()
        des_tensor=self.description_embedding()
        tweets_tensor=self.tweets_embedding()
        num_prop=self.num_prop_preprocess()
        category_prop=self.cat_prop_preprocess()
        edge_index,edge_type=self.Build_Graph()
        train_idx,val_idx,test_idx=self.train_val_test_mask()
        return des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx


    def extract_df_tweet(self):
        print('extracting df_tweet')
        df_tweet = self.df_users[self.df_users.tweet.notnull()].apply(lambda x: pd.Series([{"ID": "T"+ str(x['ID']) + "-" + str(i), "Body": tweet, **self.extractNeighborUnameDict(tweet), "tweeterId": x['ID']} for (i,tweet) in enumerate(x['tweet'])]), axis=1)
        stacked_df_tweet = df_tweet.stack()
        df_tweet = pd.DataFrame(list(stacked_df_tweet),index=pd.RangeIndex(self.df_users.shape[0], stacked_df_tweet.shape[0] + self.df_users.shape[0], 1))
        df_tweet['tweetIndex'] = df_tweet.index
        # print('df_tweet extracted')
        # print('df_tweet.shape:',df_tweet.shape)
        # print(df_tweet.columns)

        # for i,tweet in enumerate(df_tweet):
        #     print(tweet.mentions)
        #     # print(df_tweet.iloc[i,:])
        #     print('\n')
        #     break

        return df_tweet

    def extractNeighborUnameDict(self,tweet):
    # extract retweet uname

        retweet_uname = re.findall(r'(?<=RT @)(\w{1,15})', tweet)

        # extract mention unames without retweets
        mentions_no_rt = re.findall(r'(?<!RT @)(?<=@)(\w{1,15})', tweet)

        return {"retweeted": retweet_uname, "mentions": mentions_no_rt}
    
    

