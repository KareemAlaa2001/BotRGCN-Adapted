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

    def __init__(self, root='./Data/TwibotSmallAugmentedTSVDHomogeneous/',device='cpu', process=True, save=True, dev=True):
        super(TwibotSmallAugmentedTSVDHomogeneous, self).__init__(root=root,device=device, process=process, save=save, dev=dev)
    