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

from .preprocessing import preprocess_category
from sklearn.preprocessing import OrdinalEncoder


def age_map(x: int) -> int:
    x = int(x)
    if x < 10:
        return 1
    elif x >= 10 and x < 20:
        return 2
    elif x >= 20 and x < 30:
        return 3
    elif x >= 30 and x < 40:
        return 4
    elif x >= 40 and x < 50:
        return 5
    elif x >= 50 and x < 60:
        return 6
    else:
        return 7

def publish_year_map(x):
    try:
        x = int(x)
        if x < 1900:
            return 0
        else:
            return x // 10 - 190  # 1900: 0, 1990: 9, 2000: 10
    except:
        return x 

def country_map(x):
    if x in ['unitedstatesofamerica','losestadosunidosdenorteamerica','us']:
        return 'usa'
    elif x in ['england','uk']:
        return 'unitedkingdom'
    else :
        return x

def process_boosting_data(users, books, ratings1, ratings2,b_preprocess_category):
    tmp_idx = users[users['age'] > 85].index
    users.drop(tmp_idx,inplace = True)

    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users['location_country'] = users['location_country'].apply(country_map) 
    users = users.drop(['location'], axis=1)

    books['language'].fillna('en',inplace=True)

    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])

    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)
    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    # Apply ordinal encoding on language_code to convert it into numerical column
    enc = OrdinalEncoder()
    enc.fit(books[['language']])
    books[['language']] = enc.fit_transform(books[['language']]) 

    author_dict = books.groupby('book_author').count()['isbn'].sort_values()
    books['book_author'] =books['book_author'].map(author_dict)

    for publisher in modify_list:
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    publish_dict = books.groupby('publisher').count()['isbn'].sort_values()
    books['publisher'] =books['publisher'].map(publish_dict)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')

    # hhj category language 결측치 제거 
    # train_df = train_df[~(train_df['category'].isna() & train_df['language'].isna())]

    # 인덱싱 처리
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    # author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}

    if( False == b_preprocess_category ):
        train_df['category'] = train_df['category'].map(category2idx)
        test_df['category'] = test_df['category'].map(category2idx)
    else:
        train_df = preprocess_category(train_df)
        test_df = preprocess_category(test_df)

    # train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    # train_df['book_author'] = train_df['book_author'].map(author2idx)

    # test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    # test_df['book_author'] = test_df['book_author'].map(author2idx)

    # year of publication 추가
    train_df['year_of_publication'] = train_df['year_of_publication'].apply(publish_year_map)
    test_df['year_of_publication'] = test_df['year_of_publication'].apply(publish_year_map)

    idx = {
        "loc_country2idx":loc_country2idx,
        # "publisher2idx":publisher2idx,
        # "author2idx":author2idx,
    }

    return idx, train_df, test_df

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

    idx, context_train, context_test = process_boosting_data(users, books, train, test, True)

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
