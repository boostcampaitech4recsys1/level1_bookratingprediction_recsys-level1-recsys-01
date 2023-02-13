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


############################################################
# TF-IDF 시도 
############################################################
from textblob import TextBlob
import nltk 
nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.snowball import RussianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download('stopwords')

MAX_TFIDF_FEATURES = 2000
stop_words = stopwords.words('russian')
rs = RussianStemmer()

def tfidf_vectorize(series, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    return np.array(vectorizer.fit_transform(series).todense(), dtype=np.float16)

def feature_engineering(df):
    txt_vectors = tfidf_vectorize(df['concat_txt'], MAX_TFIDF_FEATURES)

    for i in range(MAX_TFIDF_FEATURES):
        df.loc[:, 'concat_txt_tfidf_' + str(i)] = txt_vectors[:, i]
    df.drop("concat_txt", axis = 1, inplace = True)
    return df

def process_boosting_data(users, books, ratings1, ratings2):
    """
    해당 함수에서는 결측치 처리 및 전처리만 한다.
    그 외 데이터 타입 변형은 after_processing 에서 모델 별로 진행 
    """

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
    # location 처리 : country 이상 데이터 통합 ( us, england 만 처리됨 )
    users = process_location_v2( target_data = users, process_level = 3 )
    ##########################################################################

    ######################## books 전처리 #####################################
    # language 값을 numerical column 으로 변환하기 위해 ordinal encoding 적용
    # [주의] category data 는 train_df, test_df 분할 후 후처리 
    # [주의] train, test 데이터 분리 후 두 명 이상의 user 가 평가한 author 에 대해 user cnt 추가  
    # [비활성화] categorical 변수를 사용하기 위해 비활성화 
    # enc = OrdinalEncoder()
    # enc.fit(books[['language']])
    # books[['language']] = enc.fit_transform(books[['language']]) 

    # books rating count : title 과 author 이 같으면 같은 책으로 봄 
    books = get_books_with_rating_count(books, ['book_title','book_author'])

    # publisher 전처리 진행 
    books = preprocess_publisher(books)
    """
    전처리된 publisher 를 count 값으로 대체할 수 있다. 
    - books['publisher'] = get_cnt_series_by_column(books,'publisher','isbn')
    - publisher_mean =  books['publisher'].mean()

    전처리된 author 를 count 값으로 대체할 수 있다. 
    - books['book_author'] = get_cnt_series_by_column(books,'book_author','isbn')
    - author_mean =  books['book_author'].mean()
    """

    # year of publication 추가
    books = process_year_of_publication(books)

    # book_rating_df = ratings1.merge(books[['isbn', 'book_author', 'book_title']], on='isbn', how='left')
    # book_rating_info = book_rating_df.groupby(["book_author","book_title"])['rating'].agg(['count','mean'])
    # book_rating_info.columns = ['rated_count','rating_mean']
    # book_rating_info.drop(book_rating_info[book_rating_info['rated_count']<50].index,inplace=True)
    # books = books.merge(book_rating_info, on=['book_author','book_title'], how='left')
    # books['rating_mean'].fillna(6.9,inplace=True)
    # books['rating_mean'] = books['rating_mean'].apply(lambda x: int(x))
    ##########################################################################

    ############################################################
    # TF-IDF 시도 
    # text preparation
    books["summary"].fillna("",inplace=True)
    books["concat_txt"] = books["book_title"].str.cat(books["summary"],sep=' ') 
    print(books[books["book_title"].isna()])
    print(books[books["concat_txt"].isna()])
    print(books[books['isbn']==129774]['summary'])
    # lower everything
    books["concat_txt"] = books["concat_txt"].str.lower() 
    # remove punctuation
    books["concat_txt"] = books["concat_txt"].str.replace('[^\w\s]',' ')

    # remove stopwords
    books["concat_txt"] = books["concat_txt"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

    # books["book_author"].fillna("",inplace=True)
    # books["concat_txt"] = books["book_title"].str.cat(books["book_author"],sep=' ') 
    # print(books[books["book_title"].isna()])
    # print(books[books["concat_txt"].isna()])
    # print(books[books['isbn']==129774]['summary'])
    # # lower everything
    # books["concat_txt"] = books["concat_txt"].str.lower() 
    # # remove punctuation
    # books["concat_txt"] = books["concat_txt"].str.replace('[^\w\s]',' ')

    # # remove stopwords
    # books["concat_txt"] = books["concat_txt"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    ############################################################

    # 인덱싱 처리된 데이터 조인
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)
    whole_df = ratings.merge(users, on='user_id', how='left').merge(books[['concat_txt','isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['concat_txt','isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['concat_txt','isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')


    # 인덱싱 처리
    """
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}

    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    """
    # category 전처리 
    train_df = preprocess_category(train_df)
    test_df = preprocess_category(test_df)

    # 작가별 단골 
    train_df = add_regular_custom_by_author(train_df)
    test_df = add_regular_custom_by_author(test_df)

    # 나이 범주화 및 결측치 채우기
    train_df = process_age( train_df , 'mean' )
    test_df = process_age( test_df , 'mean' )

    ############### 임시 결측치 처리 코드 ######################
    train_df['language'] = train_df['language'].fillna('en')
    train_df['publisher'] = train_df['publisher'].fillna(('others'))
    train_df['category'] = train_df['category'].fillna('fiction')
    train_df['book_author'] = train_df['book_author'].fillna('others')
    train_df['location_country'].fillna('na',inplace=True)
    train_df['location_state'].fillna('na',inplace=True) 
    train_df['location_city'].fillna('na',inplace=True)
    train_df['age'].fillna(4,inplace=True)
    train_df['location'].fillna('na,na,na',inplace=True)

    test_df['language'] = test_df['language'].fillna('en')
    test_df['publisher'] = test_df['publisher'].fillna(('others'))
    test_df['category'] = test_df['category'].fillna('fiction')
    test_df['book_author'] = test_df['book_author'].fillna('others')
    test_df['location_country'].fillna('na',inplace=True)
    test_df['location_state'].fillna('na',inplace=True) 
    test_df['location_city'].fillna('na',inplace=True)
    test_df['age'].fillna(4,inplace=True)
    test_df['location'].fillna('na,na,na',inplace=True)
    ###########################################################

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    return whole_df,train_df, test_df

def after_preprocessing(args, train, test, whole_df):
    
    # todo preprocess mode 에 따라 변경해도 좋을 것같다.
    if( 'CatBoost' in args.MODEL ):

        pass
        # like_books = train.groupby(['user_id'])['book_author','publisher'].value_counts().sort_values(ascending = False).reset_index()
        # like_books.columns = ['user_id','book_author', 'publisher','combine_like_book_cnt']
        # like_books['combine_like_book'] = like_books['book_author'].str.cat(like_books['publisher'],sep=',')
        # like_books.drop(['book_author','publisher'],axis=1, inplace = True )
        # like_books.drop_duplicates(['user_id'],inplace = True)

        # train= train.merge(like_books,on='user_id', how ='left')
        # test= test.merge(like_books,on='user_id', how ='left')

        # train['combine_like_book'].fillna("",inplace=True)
        # test['combine_like_book'].fillna("",inplace=True)
        # train['combine_like_book_cnt'].fillna(0,inplace=True)
        # test['combine_like_book_cnt'].fillna(0,inplace=True)
        # train['combine_like_book_cnt'] = train['combine_like_book_cnt'].astype('str')
        # test['combine_like_book_cnt'] = test['combine_like_book_cnt'].astype('str')
        
        # train.drop(['location'],axis=1, inplace = True )
        # test.drop(['location'],axis=1, inplace = True )
        
        pass 
    elif( 'XGB' in args.MODEL or 'LightGBM' in args.MODEL ):

        loc_country2idx = {v:k for k,v in enumerate(whole_df['location_country'].unique())}
        loc_state2idx = {v:k for k,v in enumerate(whole_df['location_state'].unique())}
        loc_city2idx = {v:k for k,v in enumerate(whole_df['location_city'].unique())}
        loc2idx = {v:k for k,v in enumerate(whole_df['location'].unique())}
        title2idx = {v:k for k,v in enumerate(whole_df['book_title'].unique())}

    
        train['location_country'] = train['location_country'].map(loc_country2idx)
        test['location_country'] = test['location_country'].map(loc_country2idx)
        train['location_state'] = train['location_state'].map(loc_state2idx)
        test['location_state'] = test['location_state'].map(loc_state2idx)
        train['location_city'] = train['location_city'].map(loc_city2idx)
        test['location_city'] = test['location_city'].map(loc_city2idx)
        train['location'] = train['location'].map(loc2idx)
        test['location'] = test['location'].map(loc2idx)
        train['book_title'] = train['book_title'].map(loc2idx)
        test['book_title'] = test['book_title'].map(loc2idx)

        train['publisher'] = get_cnt_series_by_column(train,'publisher','isbn')
        test['publisher'] = get_cnt_series_by_column(test,'publisher','isbn')
        train['book_author'] = get_cnt_series_by_column(train,'book_author','isbn')
        test['book_author'] = get_cnt_series_by_column(test,'book_author','isbn')

        enc = OrdinalEncoder()
        enc.fit(pd.DataFrame(pd.concat([whole_df['language']]).unique()).reset_index())
        train[['language']] = enc.fit_transform(train[['language']]) 
        test[['language']] = enc.fit_transform(test[['language']]) 

        train = map_category_with_ranking(train)
        test = map_category_with_ranking(test)

        for col in train.columns:
            if( 'object' == train[col].dtypes):
                train[col] = train[col].astype('str')

        for col in test.columns:
            if( 'object' == test[col].dtypes):
                test[col] = test[col].astype('str')

    cat_features = [f for f in train.columns if train[f].dtype == 'object' or train[f].dtype == 'category']

    return cat_features, train, test

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

    whole,boosting_train, boosting_test = process_boosting_data(users, books, train, test)
    cur_cat_features , train, test = after_preprocessing(args,boosting_train,boosting_test,whole)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'cat_features':cur_cat_features
            }


    return data


def boosting_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
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
