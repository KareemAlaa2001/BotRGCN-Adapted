import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import os
# from transformers import pipeline
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm

class TwibotSmallTruncatedSVD(Dataset):
    def __init__(self,root='./Data/TruncSVDSmall/',device='cpu',process=True,save=True,dev=False, svdComponents=100):
        self.root = root[:-1]+'Dev/' if dev else './Data/TruncSVDSmall/'
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
                self.df_data_labeled=pd.concat([df_train,df_dev,df_test],ignore_index=True)
                self.df_data=pd.concat([df_train,df_dev,df_test],ignore_index=True)
            else:
                self.df_data_labeled=df_dev
                self.df_data=df_dev
            

    def load_labels(self):
        print('Loading labels...',end='   ')
        path=self.root+'label.pt'
        if not os.path.exists(path):
            labels=torch.LongTensor(self.df_data_labeled['label']).to(self.device)
            if self.save:
                torch.save(labels,self.root+'label.pt')
        else:
            labels=torch.load(self.root+"label.pt").to(self.device)
        print('Finished')
        
        return labels

    def preprocess_descriptions(self):
        print('Loading raw descriptions',end='   ')
        path=self.root+'descriptions.npy'
        if not os.path.exists(path):
            description=[]
            for i in range (self.df_data.shape[0]):
                if self.df_data['profile'][i] is None or self.df_data['profile'][i]['description'] is None:
                    description.append('None')
                else:
                    description.append(self.df_data['profile'][i]['description'])
            description=np.array(description)
            if self.save:
                np.save(path,description)
        else:
            description=np.load(path,allow_pickle=True)
        print('Finished')
        return description

    def Des_embbeding(self):
        print('Running description embedding')
        path=self.root+"des_tensor.pt"
        if not os.path.exists(path):
            description=np.load(self.root+'descriptions.npy',allow_pickle=True)
            print('Size of descriptions matrix:',description.shape)
            print('loading tf-idf + truncated SVD')
            print('extracting tf-idf matrix')
            vectorizer = TfidfVectorizer()
            tf_idf_matrix = vectorizer.fit_transform(description)
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

    # def tweets_preprocess(self):
    #     print('Loading tweet features...',end='   ')
    #     path=self.root+'tweets.npy'
    #     if not os.path.exists(path):
    #         # tweets=[]
    #         # for i in range (self.df_data.shape[0]):
    #         #     one_usr_tweets=[]
    #         #     if self.df_data['tweet'][i] is None:
    #         #         one_usr_tweets.append('')
    #         #     else:
    #         #         for each in self.df_data['tweet'][i]:
    #         #             one_usr_tweets.append(each)
    #         #     tweets.append(one_usr_tweets)
    #         # tweets=np.array(tweets)
    #         tweets = pd.DataFrame(self.df_data['tweet'].apply(lambda tweets: [''] * 200 if tweets is None else self.pad_list_out_to_length(tweets, 200)).values.tolist()).values
    #         if self.save:
    #             np.save(path,tweets)
    #     else:
    #         tweets=np.load(path,allow_pickle=True)
    #     print('Finished')
    #     return tweets
    
    def tweets_embedding(self):
        print('Running tweet embedding')
        path=self.root+"tweets_tensor.pt"
        if not os.path.exists(path):
            maxLength = self.df_data['tweet'].apply(lambda tweets: len(tweets) if tweets is not None else 0).max()
            tweets = pd.DataFrame(self.df_data['tweet'].apply(lambda tweets: [''] * maxLength if tweets is None else self.pad_list_out_to_length(tweets, maxLength)).values.tolist()).values
            # print('Loading RoBerta')
            # print('current device value', self.device_value)
            # feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base',device=self.device_value,padding=True, truncation=True,max_length=500, add_special_tokens = True)
            
            print('Size of tweets matrix:',tweets.shape)
            print('type of elements in tweets matrix:', tweets.dtype)
            print('tweets matrix first element', tweets[0])

            print('loading tf-idf + truncated SVD')
            print('extracting tf-idf matrix')
            vectorizer = TfidfVectorizer()

            print('concatenating tweets into one nparray')
            npTweetsConcat = np.concatenate(tweets)
            print('fitting df-idf vectoriser')
            vectorizer.fit(npTweetsConcat)

            print('transforming tweets to tf-idf matrix')
            tf_idf_matrix = np.array([vectorizer.transform(usrtweets) for usrtweets in tqdm(tweets)])


            print("tf-idf matrix dimensions:", tf_idf_matrix.shape)
            print('fitting truncated SVD')
            svd = TruncatedSVD(n_components=self.svdComponents, n_iter=5, random_state=42) # n_iter=5 is the default, n_components=2 is the default
            svd.fit(vstack(tf_idf_matrix))
            user_tweet_svds = torch.Tensor([svd.transform(usertweettfidf) for usertweettfidf in tqdm(tf_idf_matrix)])
            print('user_tweet_svds shape:', user_tweet_svds.shape)
            print('datatype in user_tweet_svds:', user_tweet_svds.dtype)
            averaged_tweet_embeddings = torch.mean(user_tweet_svds, axis=1)
            # trunc_svd_matrix = svd.fit_transform(csr_tfidf_matrix)
            # print('dimensionality of tweets truncated svd matrix:',trunc_svd_matrix.shape)
            
            tweets_tensor=averaged_tweet_embeddings.to(self.device)
            if self.save:
                torch.save(tweets_tensor,path)
        else:
            tweets_tensor=torch.load(path).to(self.device)
        print('Finished')
        return tweets_tensor
    
    def num_prop_preprocess(self):
        print('Processing feature3...',end='   ')
        path0=self.root+'num_prop.pt'
        if not os.path.exists(path0):
            path=self.root
            if not os.path.exists(path+"followers_count.pt"):

                numerical_feature_names = ['followers_count', 'friends_count','favourites_count','statuses_count']
                followers_count, friends_count, favourites_count, statuses_count = [self.extractNumericalFeatureFromDf(feature_name, self.df_data).to(self.device) for feature_name in numerical_feature_names]
                print('typeof followers_count:',type(followers_count))
                print('followers_count shape:', followers_count.shape)
                if self.save:
                    for feature_name in numerical_feature_names:
                        torch.save(feature_name,path+feature_name+'.pt') 
                    
                ## TODO handle this separately from the other classes being cleaned up
                screen_name_length=[]
                for i in range (self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['screen_name'] is None:
                        screen_name_length.append(0)
                    else:
                        screen_name_length.append(len(self.df_data['profile'][i]['screen_name']))
                screen_name_length=torch.tensor(np.array(screen_name_length,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(screen_name_length,path+'screen_name_length.pt')
                
                ## TODO handle this separately from the other classes being cleaned up
                active_days=[]
                date0=dt.strptime('Fri Jul 1 00:00:00 +0000 2022 ','%a %b %d %X %z %Y ')
                for i in range (self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['created_at'] is None:
                        active_days.append(0)
                    else:
                        date=dt.strptime(self.df_data['profile'][i]['created_at'],'%a %b %d %X %z %Y ')
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

            num_prop=torch.cat((followers_count.reshape([self.df_data.shape[0],1]),friends_count.reshape([self.df_data.shape[0],1]),favourites_count.reshape([self.df_data.shape[0],1]),statuses_count.reshape([self.df_data.shape[0],1]),screen_name_length_days.reshape([self.df_data.shape[0],1]),active_days.reshape([self.df_data.shape[0],1])),1).to(self.device)

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
            for i in range (self.df_data.shape[0]):
                prop=[]
                if self.df_data['profile'][i] is None:
                    prop = [0] * len(properties)
                else:
                    for each in properties:
                        if self.df_data['profile'][i][each] is None:
                            prop.append(0)
                        else:
                            if self.df_data['profile'][i][each] == "True ":
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
            id2index_dict={id:index for index,id in enumerate(self.df_data['ID'])}
            edge_index=[]
            edge_type=[]
            for i,relation in enumerate(self.df_data['neighbor']):
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
                        edge_type.append(1)
                else:
                    continue
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
    
    def train_val_test_mask(self):
        if not self.dev:
            train_idx=range(8278)
            val_idx=range(8278,8278+2365)
            test_idx=range(8278+2365,8278+2365+1183)
        
        else:
            train_idx = range(int(0.8*len(self.df_data)))
            val_idx = range(int(0.8*len(self.df_data)),int(0.9*len(self.df_data)))
            test_idx = range(int(0.9*len(self.df_data)),len(self.df_data))

        return train_idx,val_idx,test_idx

    def dataloader(self):
        labels=self.load_labels()
        self.preprocess_descriptions()
        des_tensor=self.Des_embbeding()
        # self.tweets_preprocess()
        tweets_tensor=self.tweets_embedding()
        num_prop=self.num_prop_preprocess()
        category_prop=self.cat_prop_preprocess()
        edge_index,edge_type=self.Build_Graph()
        train_idx,val_idx,test_idx=self.train_val_test_mask()
        return des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx

    def extractNumericalFeatureFromDf(self, feature: str, df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor([0 if (df['profile'][i] is None or df['profile'][i][feature] is None) else float(df['profile'][i][feature]) for i in range(df.shape[0])], dtype=torch.float32)

    def pad_list_out_to_length(self, list_out, length: int):
        if len(list_out) < length:
            list_out.extend([''] * (length - len(list_out)))
        return list_out