from TwibotSmallAugmentedTSVDHomogeneous import TwibotSmallAugmentedTSVDHomogeneous
import torch
import os
import pandas as pd

class TwibotSmallEdgeHetero(TwibotSmallAugmentedTSVDHomogeneous):

    def Build_Graph(self):
        print('Building graph',end='   ')
        path=self.root+'hetero_edge_index.pt'
        if not os.path.exists(path):
            if self.df_tweet == None:
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
                        edge_type.append(1)
                else:
                    continue

            # building the edges from the tweet relationships

            for mentions, retweeted, tweeterid, tweetIndex in zip(self.df_tweet['mentions'], self.df_tweet['retweeted'], self.df_tweet['tweeterId'], self.df_tweet['tweetIndex']):
                for mention in mentions:
                    if uname2id_dict.get(mention) is not None:
                        if id2index_dict.get(int(uname2id_dict[mention])) is not None:
                            edge_index.append([int(tweetIndex),id2index_dict[int(uname2id_dict[mention])]])
                            edge_type.append(2)

                for rt in retweeted:
                    if uname2id_dict.get(rt) is not None:
                        if id2index_dict.get(int(uname2id_dict[rt])) is not None:
                            target_index= id2index_dict[int(uname2id_dict[rt])]
                            edge_index.append([int(tweetIndex),target_index])
                            edge_type.append(3)
        
                if id2index_dict.get(int(tweeterid)) is not None:
                    target_index=id2index_dict[int(tweeterid)]
                    edge_index.append([target_index,int(tweetIndex)])
                    edge_type.append(4)
               
            edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(self.device)
            edge_type=torch.tensor(edge_type,dtype=torch.long).to(self.device)
            if self.save:
                torch.save(edge_index,self.root+"hetero_edge_index.pt")
                torch.save(edge_type,self.root+"hetero_edge_type.pt")
        else:
            edge_index=torch.load(self.root+"hetero_edge_index.pt").to(self.device)
            edge_type=torch.load(self.root+"hetero_edge_type.pt").to(self.device)
            print('Finished')
        return edge_index,edge_type