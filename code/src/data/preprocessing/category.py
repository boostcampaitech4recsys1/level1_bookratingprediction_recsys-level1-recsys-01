import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re

# 입력으로 전달되는 category x 값을 전처리한다.

def isNaN(num):
    return num != num

def get_category_high(x):
    if  isNaN(x):
        return x
    elif 'language' in x : 
        return 'linguistics'
    elif 'art' in x or 'craftshobbies' in x or 'photography' in x:
        return 'art'
    elif 'nonfiction' in x or 'true crime' in x:
        return 'nonfiction'
    elif 'novel' in x or 'act' in x or'fictitious' in x or 'fiction' in x or 'stories' in x or 'drama' in x or 'children' in x :
        return 'fiction'
    elif 'adventure' in x :
        return 'adventure'
    elif 'biography' in x:
        return 'biography'
    elif 'history' in x:
        return 'history'
    elif 'religion' in x or 'bible' in x :
        return 'bible'
    elif 'humor' in x or 'comic' in x or 'comics' in x:
        return 'comic'
    elif 'child' in x or 'mind' in x or 'psychology' in x:
        return 'psychology'
    elif 'business' in x:
        return 'business'
    elif 'cook' in x:
        return 'food'
    elif 'health' in x or 'mind' in x:
        return 'health'
    elif 'business' in x:
        return 'business'
    elif 'relationships' in x:
        return 'sociology'
    elif 'computer' in x or 'technology' in x or 'engineer' in x or 'electron' in x:
        return 'technology'
    elif 'game' in x:
        return 'games'
    elif 'architect' in x or 'gardening' in x :
        return 'architecture'
    elif 'pet' in x or 'cat' in x or 'animal' in x :
        return 'pets'
    elif 'sport' in x:
        return 'sports'
    elif 'political science' in x:
        return 'socialogy'
    elif 'education' in x or 'study' in x:
        return 'socialogy'
    elif 'crim' in x:
        return 'criminal'
    elif 'philosophy' in x :
        return 'philosophy'
    elif 'travel' in x :
        return 'travel'
    elif 'science' in x:
        return 'science'
    elif 'animat' in x:
        return 'animation'
    else :
        return x

def map_category_with_ranking( target_data ):
    
    """
    preprocess_category 를 통해 전처리된 category 가 전달 되어야 한다.
    """
    target_data['category'] = target_data['category'].map({'fiction' : 1, 'juvenilefiction' : 1, 'biography' : 2, 'comic' : 3,'psychology': 4,
                                      'history' : 5, 'bible' : 6, 'nonfiction' : 7, 'science' : 8, 'art' : 9, 'business' : 10})
    # 나머지는 1으로 채워준다. 
    target_data['category'] = target_data['category'].fillna(1)
    return target_data


def preprocess_category( target_data , threshold_cnt = 5):
    
    # category column 의 string 전처리 ( 특수문자 삭제 및 소문자 변환 )는 이미 된 상태로 전달 되어야 함 
    target_data['category_high'] = target_data['category'].apply(get_category_high) 
    # 나머지는 'fiction' 으로 채워준다. 
    target_data['category_high'].fillna('fiction')

    category_high_df = pd.DataFrame(target_data['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    
    # threshold 미만인 것들을 fiction 으로 대입해준다.
    others_list = category_high_df[category_high_df['count'] < threshold_cnt]['category'].values
    target_data.loc[target_data[target_data['category_high'].isin(others_list)].index, 'category_high']='fiction'
    target_data['category'] = target_data['category_high']
    target_data = target_data.drop(columns = ['category_high']) 

    return target_data
    