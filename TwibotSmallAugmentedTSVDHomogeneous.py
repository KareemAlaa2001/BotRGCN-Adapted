import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# import json
# import os
# from datetime import datetime as dt
# from torch.utils.data import Dataset
# from tqdm import tqdm
from TwibotSmallTruncatedSVD import TwibotSmallTruncatedSVD

class TwibotSmallAugmentedTSVDHomogeneous(TwibotSmallTruncatedSVD):

    def __init__(self, root='./Data/TwibotSmallAugmentedTSVDHomogeneous/',device='cpu', process=True, save=True, dev=False, svdComponents=100):
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
    

