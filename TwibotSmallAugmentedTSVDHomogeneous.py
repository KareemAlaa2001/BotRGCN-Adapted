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

class TwibotSmallTruncatedSVD(Dataset):
    def __init__(self,device,process=True,save=True,dev=False):
        self.device=device
        self.process=process
        self.save=save
        self.dev=dev

    pass ## TODO