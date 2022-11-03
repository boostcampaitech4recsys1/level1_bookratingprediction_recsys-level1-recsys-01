import pandas as pd
from . import get_apply_map_series

def get_books_with_rating_count(books:pd.DataFrame, orient_column:list):
    
    books_ranking = books.groupby(orient_column)['isbn'].count().sort_values(ascending =False).reset_index()
    new_column = orient_column.copy()
    new_column.append('book_rating_cnt')
    books_ranking.columns = new_column
    merged_books = books.merge(right = books_ranking, on = orient_column , how = 'left')
    
    return merged_books

def publish_year_map(x):

    # publish_year 에 null 값은 없다고 가정한다. 
    x = int(x)
    if x < 1900:
        return 0
    else:
        return x // 10 - 190  # 1900: 0, 1990: 9, 2000: 10


def preprocess_publisher( target_data:pd.DataFrame)->pd.DataFrame:

    # if( 'publisher' not in target_data):
    #     print('[WARN][remove_special_char_of_str] ', key, ' is not element of target_data')
    #     return None 

    # if( 'isbn' not in target_data):
    #     print('[WARN][remove_special_char_of_str] The primary key isbn is not element of target_data')
    #     return None 

    # publisher_dict=(target_data['publisher'].value_counts()).to_dict()
    # publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])

    # publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)
    # # 2개 이상인 출판사를 정제 대상으로 한다. 
    # modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

    # for publisher in modify_list:
    #     try:
    #         # 처리 대상인 출판사의 첫 네 글자를 대상으로 가장 많이 나온 출판사 명칭으로 바꾸어준다. 
    #         number = target_data[target_data['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
    #         right_publisher = target_data[target_data['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
    #         target_data.loc[target_data[target_data['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
    #     except: 
    #         pass

    return target_data

def process_year_of_publication(target_data : pd.DataFrame )->pd.DataFrame :
    
    if( 'year_of_publication' not in target_data):
        print('[WARN][process_year_of_publication] year_of_publication is not element of target_data')
        return None

    target_data['year_of_publication'] = get_apply_map_series(target_data,'year_of_publication',publish_year_map)
    return target_data 

def process_series(target_data : pd.DataFrame )->pd.DataFrame :
    series = target_data.groupby(['book_author', 'publisher', 'category'])[['book_title']].count().rename(columns={'book_title': 'series_cnt'}).reset_index()
    series = series[series['series_cnt'] > 1]
    series['series'] = series['book_author'].astype(str)+'-'+series['publisher'].astype(str)+'-'+series['category'].astype(str)
    target_data = target_data.merge(series, on=['book_author', 'publisher', 'category'], how='left')
    target_data['series'].fillna('others', inplace=True)
    return target_data