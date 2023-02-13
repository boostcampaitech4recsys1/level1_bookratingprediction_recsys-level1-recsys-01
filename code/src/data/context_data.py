import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import re
from torch.utils.data import TensorDataset, DataLoader, Dataset

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .preprocessing import *
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
    if x in ['','na']:
        return 'usa'
    
    if x in ['unitedstatesofamerica','losestadosunidosdenorteamerica','us','unitedstate','unitedstaes','unitedstatesofamerica','unitedsates','unitedstates']:
        return 'usa'
    elif x in ['england','uk','unitedkingdom','unitedkindgonm']:
        return 'unitedkingdom'
    elif x in ['hongkong']:
        return 'china'
    elif x in ['deutschland']:
        return 'germany'
    elif x in ['catalunya','espaa']:
        return 'spain'
    else :
        return x

# def sep_country(x):
#     if x in ['finland','portugal','germany','austria','italy','france','netherlands','poland','spain','singapore','switzerland','yugoslavia','sweden','slovakia','norway','macedonia','denmark','russia','andorra','czechrepublic','bulgaria','slovenia','luxembourg','iceland','cyprus','albania','ukraine','malta','romania','greece','ireland','belgium','unitedkingdom','argentina','hungary','croatia','lithuania']:
#         return 'Europe'
#     elif x in ['iran','turkey','qatar','kuwait','israel','saudiarabia','unitedarabemirates']:
#         return 'MiddleEast'
#     elif x in ['malaysia','taiwan','singapore','philippines','japan','china','indonesia','nepal','southkorea','pakistan','srilanka','india','turkmenistan','laos','thailand']:
#         return 'Asia'
#     elif x in ['southafrica', 'kenya','mauritius','madagascar','egypt','zambia','zimbabwe','algeria','mozambique','guinea','canaryislands','nigeria']:
#         return 'Africa'
#     elif x in ['newzealand','australia','palau']:
#         return 'Oceania'
#     elif x in ['canada','usa','bermuda','puertorico']:
#         return 'NorthAmerica'
#     elif x in ['brazil', 'costarica', 'mexico', 'peru','dominicanrepublic','guatemala','venezuela','chile','honduras','trinidadandtobago','belize','jamaica','bahamas','caymanislands','saintlucia','barbados']:
#         return 'SouthAmerica'
#     elif x in ['quit','universe','tdzimi','stthomasi','faraway','everywhereandanywhere','ontario','k1c7b1','naontheroad','sthelena']:
#         return 'Remove_Point'
#     else:
#         return x

def process_context_data(users, books, ratings1, ratings2, b_preprocess_category):
    # ===================== 1. users preprocessing =====================
    # ===================== 1-1. location
    #users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    #users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    # users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[-1])
    # 공백 및 특수문자 제거
    users['location_country'] = users['location_country'].apply(lambda x: re.sub('[^0-9a-zA-Z]','',x).strip())
    users = users.drop(['location'], axis=1)

    # gu: location_country 결측치 처리
    users['location_country'] = users['location_country'].fillna('usa')
    users['location_country'] = users['location_country'].apply(country_map)

    # 6대주 + 중동

    # users['location_country'] = users['location_country'].apply(sep_country)

    # 해당 나라의 유저가 5명 이하일 경우 해당 나라 제거
    country_unique = users['location_country'].unique().tolist()
    for country in country_unique:
        if users[users['location_country']==country].shape[0]< 5:
            users.loc[users['location_country']==country,"location_country"] = 'Remove_Point'
    users.drop(users[users['location_country']=='Remove_Point'].index,inplace=True)


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
    # books['publisher'] = books['publisher'].fillna('others') # 퍼블리셔 결측치 없음


    # ===================== 2-2. language
    # # Apply ordinal encoding on language_code to convert it into numerical column
    # enc = OrdinalEncoder()
    # enc.fit(books[['language']])
    # books[['language']] = enc.fit_transform(books[['language']])

    # # hhj category language 결측치 제거 
    # train_df = train_df[~(train_df['category'].isna() & train_df['language'].isna())]

def process_context_data(users, books, ratings1, ratings2, b_preprocess_category):

    # ===================== 0. string type preprocessing =====================
    books = process_str_column(['language', 'category','book_author','publisher'],books )
    users = process_str_column(['location'],users )

    # ===================== 1. users preprocessing =====================
    # ===================== 1-1. location
    # location 처리 - country, state, city 처리 및 country 이상 데이터 통합 ( us, england 만 처리됨 )
    # gu: location_country 결측치 처리
    users = process_location( target_data = users, process_level = 3 )
    users['location_country'] = users['location_country'].fillna('usa')

    # ===================== 1-2. age
    users = process_age( users , 'mean' )

    # ===================== 2. books preprocessing =====================
    # ===================== 2-1. publisher
    books = preprocess_publisher(books)
    # gu: publisher 결측치 처리
    books['publisher'] = books['publisher'].fillna('others')

    # ===================== 2-2. language
    # # hhj category language 결측치 제거 
    # train_df = train_df[~(train_df['category'].isna() & train_df['language'].isna())]

    # gu: language 결측치처리
    books['language'] = books['language'].fillna('en')

    # ===================== 2-3. author
 
    # gu: book_author 전처리
    books['book_author'] = books['book_author'].str.replace(r'[^0-9a-zA-Z:,]', '')
    
    # [주의] train, test data 분리 후 2명 이상 평가한 author count 를 추가한다.

    # ===================== 2-4. category
    books['category'] = books['category'].fillna('fiction')

    # category 전처리 위치 이동
    books = preprocess_category(books)
    books = map_category_with_ranking(books)

    # 시리즈 추가(author-publisher-category): 카테고리 전처리 후에 진행
    books = process_series(books)

    # year of publication 추가
    books = process_year_of_publication(books)


    # ===================== 3. merge and indexing =====================
    # ===================== 3-1. merge
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication', 'series']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication', 'series']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication', 'series']], on='isbn', how='left')

    # ===================== 3-2. users columns 인덱싱처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    #train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    #train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    #test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    #test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # ===================== 3-3. book author/publisher preprocessing after merge
    # gu book_author/publiser 변수 추가
    # 1) 작가별 출판 책 수 추가
    # author_cnt = books.groupby('book_author')[['isbn']].agg('count').sort_values(by='isbn', ascending=False).rename(columns={'isbn': 'author_book_cnt'}).reset_index()
    # train_df = train_df.merge(author_cnt, on='book_author', how='left')
    # test_df = test_df.merge(author_cnt, on='book_author', how='left')
    # 2) 작가별 단골 추가
    train_df = add_regular_custom(train_df, 'book_author')
    test_df = add_regular_custom(test_df, 'book_author')
    # 3) 출판사별 단골 
    train_df = add_regular_custom(train_df, 'publisher')
    test_df = add_regular_custom(test_df, 'publisher')

    # age and year of publication 갭 추가
    train_df['age_pub_gap'] = train_df['year_of_publication'] - train_df['age'] + 10  # 음수 제거용
    test_df['age_pub_gap'] = test_df['year_of_publication'] - test_df['age'] + 10  # 음수 제거용

    # ===================== 3-4. books columns 인덱싱처리
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    series2idx = {v:k for k,v in enumerate(context_df['series'].unique())}

    if( False == b_preprocess_category ):
        train_df['category'] = train_df['category'].map(category2idx)
        test_df['category'] = test_df['category'].map(category2idx)

    # else:
    #     train_df = preprocess_category(train_df)
    #     train_df = map_category_with_ranking(train_df)
    #     test_df = preprocess_category(test_df)
    #     test_df = map_category_with_ranking(test_df)

    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    train_df['series'] = train_df['series'].map(series2idx)

    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    test_df['series'] = test_df['series'].map(series2idx)

    # # gu: year of publication 추가
    # train_df = process_year_of_publication(train_df)
    # test_df = process_year_of_publication(test_df)

    # # SVD, CoClustering 값 채우기
    # print('before combine')
    # print(train_df.shape, test_df.shape, len(ratings1))
    # train_df, test_df = combine_features(ratings1, ratings2, train_df, test_df)
    # print('after combine')
    # print(train_df.shape, test_df.shape)
    # # print(train_df.head())
    # # print(ratings.head())
    # # print(ratings1.head())

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
        "series2idx":series2idx,
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
    print(context_train.head())
    print(context_train.columns)
    # 7: age 
    # 10: year_of_publication, 
    # author_book_cnt, 
    # 4: book_author_common_cnt, 
    # 5: publisher_common_cnt, 
    # age-pub gap
    # series
    field_dims = np.array([len(user2idx), len(isbn2idx), 7,
                            len(idx['loc_country2idx']), len(idx['loc_state2idx']), len(idx['loc_city2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx']),
                            10, len(idx['series2idx']), len(context_train['book_author_common_cnt'].unique()), len(context_train['publisher_common_cnt'].unique()), len(context_train['age_pub_gap'].unique())], dtype=np.uint32)
    print(field_dims)
    # field_dims = np.array([len(user2idx), len(isbn2idx),
    #                         7, 10, len( context_train['author_common_cnt']), len(context_train['author_book_cnt']), len(idx['loc_country2idx']),len(idx['loc_state2idx']),len(idx['loc_city2idx']),
    #                         len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

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