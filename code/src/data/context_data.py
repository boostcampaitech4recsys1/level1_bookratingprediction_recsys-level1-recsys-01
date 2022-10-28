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
from sklearn.preprocessing import OrdinalEncoder

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

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

def process_context_data(users, books, ratings1, ratings2, b_preprocess_category):
    # ===================== 1. users preprocessing =====================
    # ===================== 1-1. location
    #users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    #users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users['location_country'] = users['location_country'].apply(country_map) 
    users = users.drop(['location'], axis=1)

    # gu: location_country 결측치 처리
    users['location_country'] = users['location_country'].fillna('usa')


    # ===================== 2. books preprocessing =====================
    # ===================== 2-1. publisher
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])

    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)
    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    for publisher in modify_list:
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass

    # gu: publisher 결측치 처리
    books['publisher'] = books['publisher'].fillna('others')


    # ===================== 2-2. language
    # # Apply ordinal encoding on language_code to convert it into numerical column
    # enc = OrdinalEncoder()
    # enc.fit(books[['language']])
    # books[['language']] = enc.fit_transform(books[['language']])

    # # hhj category language 결측치 제거 
    # train_df = train_df[~(train_df['category'].isna() & train_df['language'].isna())]

    # gu: language 결측치처리
    books['language'] = books['language'].fillna('en')

    # ===================== 2-3. author
    # gu: book_author 전처리
    books['book_author'] = books['book_author'].str.replace(r'[^0-9a-zA-Z:,]', '')


    # ===================== 3. merge and indexing =====================
    # ===================== 3-1. merge
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')

    # ===================== 3-2. users columns 인덱싱처리
    #loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    #loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    #train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    #train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    #test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    #test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # ===================== 3-3. book author preprocessing after merge
    # gu book_author 변수 추가
    # 1) 작가별 출판 책 수 추가
    author_cnt = books.groupby('book_author')[['isbn']].agg('count').sort_values(by='isbn', ascending=False).rename(columns={'isbn': 'author_book_cnt'}).reset_index()
    train_df = train_df.merge(author_cnt, on='book_author', how='left')
    test_df = test_df.merge(author_cnt, on='book_author', how='left')
    # 2) 작가별 단골 추가
    common = train_df.groupby(['book_author', 'user_id'])[['rating']].count()
    author_common = common[common['rating']>2].groupby('book_author').count().sort_values('rating', ascending=False).rename(columns={'rating': 'author_common_cnt'}).reset_index()
    train_df = train_df.merge(author_common, on='book_author', how='left')
    train_df['author_common_cnt'].fillna(0, inplace=True)
    test_df = test_df.merge(author_common, on='book_author', how='left')
    test_df['author_common_cnt'].fillna(0, inplace=True) 

    # ===================== 3-4. books columns 인덱싱처리
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}

    if( False == b_preprocess_category ):
        train_df['category'] = train_df['category'].map(category2idx)
        test_df['category'] = test_df['category'].map(category2idx)
    else:
        train_df = preprocess_category(train_df)
        test_df = preprocess_category(test_df)

    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)

    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    # gu: year of publication 추가
    train_df['year_of_publication'] = train_df['year_of_publication'].apply(publish_year_map)
    test_df['year_of_publication'] = test_df['year_of_publication'].apply(publish_year_map)

    idx = {
        #"loc_city2idx":loc_city2idx,
        #"loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df

def context_data_load(args):

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

    idx, context_train, context_test = process_context_data(users, books, train, test, True)
    # field_dims = np.array([len(user2idx), len(isbn2idx),
    #                         6, 10, 1, 1, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
    #                         len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, 10, 1, 1, len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
