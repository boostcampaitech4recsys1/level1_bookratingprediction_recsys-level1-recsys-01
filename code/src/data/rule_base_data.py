import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .preprocessing import preprocess_category

def get_popular_rating_dict_by_column( target_df, column_name, top_n=100, unique_column ='isbn' ):
   
    if( column_name not in target_df.columns ):
        print("[Error] Not valid column name : ", column_name)
        return

    popular_data = target_df.groupby([column_name])[unique_column].count().sort_values(ascending = False)[:top_n].reset_index()
    popular_data.columns = [column_name,'cnt']
    popular_data_dict = target_df[target_df[column_name].isin(popular_data[column_name])].groupby([column_name])['rating'].mean().reset_index()
    popular_data_dict.columns = [column_name,'rating']

    return popular_data_dict.copy()


def process_basic_regrex(target_df, column_name):

    target_df[column_name] = target_df[column_name].str.replace(r'[^0-9a-zA-Z:,]','')
    target_df[column_name] = target_df[column_name].str.lower()
    return 

def process_rule_base_data(users, books, ratings1, ratings2, b_preprocess_category):
    process_basic_regrex(books,'book_author')
    books = preprocess_category(books)

    # 인덱싱 처리된 데이터 조인
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    context_df = ratings.merge(users, on='user_id', how='left').merge(books, on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books, on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books, on='isbn', how='left')

    # 인기 상품에 대한 dictionary
    popular_author_dict = get_popular_rating_dict_by_column(context_df,'book_author')
    many_reader_dict = get_popular_rating_dict_by_column(context_df,'user_id')
    popular_category_dict = get_popular_rating_dict_by_column(context_df,'category')
    many_reader_with_cat_dict =  context_df.groupby(['user_id','category_rank'])['rating'].mean().reset_index()
    many_reader_with_cat_dict.columns = ['user_id','category_rank','rating']

    popular_collections = { 'author':popular_author_dict,  
                            'reader':many_reader_dict, 
                            'category':popular_category_dict ,
                            'reader_with_category':many_reader_with_cat_dict}

    return popular_collections, train_df, test_df

def rule_base_data_load(args):

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

    # 사용할 column만 골라보자.
    # 변환된 idx 값이 id 와 isbn 대신 사용된다 .
    users = users[['user_id']]
    books = books[['isbn','book_author','category']]


    popular_collections, context_train, context_test = process_rule_base_data(users, books, train, test, True)
    
    data = {
            'popular':popular_collections,
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


def rule_base_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
   
    return data