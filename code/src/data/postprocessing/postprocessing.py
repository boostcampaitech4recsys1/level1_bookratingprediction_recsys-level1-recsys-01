import pandas as pd

def get_rating_average(submission:pd.DataFrame, process_level:int )->pd.DataFrame:
    books = pd.read_csv('/opt/ml/input/code/data/books.csv')
    users = pd.read_csv('/opt/ml/input/code/data/users.csv')
    ratings1 = pd.read_csv('/opt/ml/input/code/data/train_ratings.csv')
    rating_df = ratings1.merge(books[['isbn', 'book_author', 'book_title']], on='isbn', how='left')

    #책 평균 평점    
    books_rating_info = rating_df.groupby(["book_author","book_title"])['rating'].agg(['mean'])
    books_rating_info.columns = ['books_mean']
    books_rating_info = books.merge(books_rating_info,on=['book_title','book_author'],how='left')
    books_rating_info = books_rating_info[['isbn','books_mean']]
    books_rating_info['books_mean'] = books_rating_info['books_mean'].fillna(books_rating_info['books_mean'].mean()) # 결측치 평균으로 채움
    
    #유저 평균 평점
    users_rating_info = rating_df.groupby(["user_id"])['rating'].agg(['mean'])
    users_rating_info.columns = ['users_mean']
    users_rating_info['users_mean'] = users_rating_info['users_mean'].fillna(users_rating_info['users_mean'].mean())
    
    #merge
    rating_avg = submission.merge(books_rating_info,on=['isbn'],how='left').merge(users_rating_info,on='user_id',how='left')
    
    # 평균 연산
    
    if process_level==1: # 책 평균만
        submission['rating'] = rating_avg[['rating','books_mean','users_mean']].mean(axis=1)
    elif process_level==2:
        submission['rating'] = rating_avg[['rating','users_mean']].mean(axis=1)
    elif process_level==3: # 책 평균, 유저 평균
        submission['rating'] = rating_avg[['rating','books_mean','users_mean']].mean(axis=1)
    
    return submission