import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re

# 입력으로 전달되는 category x 값을 전처리한다.
def get_category_high(x : str ):
    if 'language arts' in x : 
        return 'linguistics'
    elif 'arts' in x or 'crafts hobbies' in x or 'photography' in x:
        return 'art'
    elif 'nonfiction' in x or 'true crime' in x:
        return 'nonfiction'
    elif 'fiction' in x or 'stories' in x or 'drama' in x or 'children' in x :
        return 'fiction'
    elif 'biography' in x:
        return 'biography'
    elif 'history' in x:
        return 'history'
    elif 'religion' in x:
        return 'bible'
    elif 'humor' in x or 'comic' in x or 'comics' in x:
        return 'comic'
    elif 'mind' in x or 'psychology' in x:
        return 'psychology'
    elif 'business' in x:
        return 'business'
    elif 'cooking' in x:
        return 'food'
    elif 'health' in x or 'mind' in x:
        return 'health'
    elif 'business' in x:
        return 'business'
    elif 'relationships' in x:
        return 'sociology'
    elif 'computers' in x or 'technology' in x or 'engineering' in x:
        return 'sociology'
    elif 'games' in x:
        return 'games'
    elif 'architecture' in x or 'gardening' in x :
        return 'architecture'
    elif 'pets' in x or 'cats' in x or 'animals' in x :
        return 'pets'
    elif 'sports' in x:
        return 'sports'
    elif 'political science' in x:
        return 'socialogy'
    elif 'education' in x or 'study' in x:
        return 'socialogy'
    elif 'criminal' in x:
        return 'criminal'
    else :
        return x

def preprocess_category( books ):
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()
    # 결측치는 우선 ohter 로 채워준다. 
    books['category'].fillna('ohter',inplace=True)
    books['category_high'] =books['category'].apply(get_category_high) 
    books['category_rank'] = books.category_high.map({'fiction' : 1, 'juvenile fiction' : 1, 'biography' : 2, 'history' : 3,'sociology': 4,
                                      'bible' : 5, 'psychology' : 6, 'nonfiction' : 7, 'comic' : 8, 'art' : 9})
    # 나머지는 10으로 채워준다. 
    books['category_rank'].fillna(10.0)
    books.drop(columns=['category','category_high'],inplace=True)
    return books