import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .preprocessing import *
from sklearn.preprocessing import OrdinalEncoder

def process_boosting_data(users, books, ratings1, ratings2,b_preprocess_category):
    # 문자열 관련 미리 처리
    books = process_str_column(['language', 'category','book_author','publisher'],books )
    users = process_str_column(['location'],users )

   ######################## 결측치 처리  #####################################
    books['language'] = books['language'].fillna('en')
    books['publisher'] = books['publisher'].fillna('others')
    books['category'] = books['category'].fillna('fiction')
    ##########################################################################

    ######################## users 전처리 #####################################
    # age 관련 처리 : 86세 이상의 데이터는 버림 처리 
    users = remove_outlier_by_age(users,85)
    users = process_age( users , 'mean' )
    # location 처리 : country 이상 데이터 통합 ( us, england 만 처리됨 )
    users = process_location( target_data = users, process_level = 3 )
    ##########################################################################

    ######################## books 전처리 #####################################
    # language 값을 numerical column 으로 변환하기 위해 ordinal encoding 적용
    # [주의] category data 는 train_df, test_df 분할 후 후처리 
    # [주의] train, test 데이터 분리 후 두 명 이상의 user 가 평가한 author 에 대해 user cnt 추가  
    enc = OrdinalEncoder()
    enc.fit(books[['language']])
    books[['language']] = enc.fit_transform(books[['language']]) 

    # books rating count : title 과 author 이 같으면 같은 책으로 봄 
    books = get_books_with_rating_count(books, ['book_title','book_author'])

    # publisher 전처리 진행 
    books = preprocess_publisher(books)
    # 전처리된 publisher 를 count 값으로 대체 
    books['publisher'] = get_cnt_series_by_column(books,'publisher','isbn')
    
    # author 를 count 값으로 대체 
    books['book_author'] = get_cnt_series_by_column(books,'book_author','isbn')

    # year of publication 추가
    books = process_year_of_publication(books)
    ##########################################################################

 
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')

    # 인덱싱 처리
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}

    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)

    # category 전처리 
    train_df = preprocess_category(train_df)
    test_df = preprocess_category(test_df)

    # 작가별 단골 
    train_df = add_regular_custom_by_author(train_df)
    test_df = add_regular_custom_by_author(test_df)

    return train_df, test_df

def boosting_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    context_train, context_test = process_boosting_data(users, books, train, test, True)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def boosting_data_split(args, data):
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    folds = []

    for train_idx, valid_idx in skf.split(data['train'].drop(['rating'], axis=1), data['train']['rating']):
        folds.append((train_idx,valid_idx))
    
    data['folds'] = folds 

    return data


def boosting_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
