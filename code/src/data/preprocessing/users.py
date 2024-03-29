import pandas as pd
import re
from . import get_apply_map_series

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
def country_map(x):
    # if x in ['','na']:
    #     return 'usa'
    # 도시명만 존재할 경우에도 나라가 usa로 바뀜.
    if x in ['ca','canada','can','k1c7b1']:
        return 'can'
    if x in ['america','everywhereandanywhere','stthomasi','csa','losestadosunidosdenorteamerica','unitedstatesofamerica','losestadosunidosdenorteamerica','us','unitedstate','unitedstaes','unitedstatesofamerica','unitedsates','unitedstates']:
        return 'usa'
    elif x in [ 'gbr','uk','unitedkingdom','unitedkindgonm']:
        return 'gbr'
    elif x in ['deutschland','germany']:
        return 'deu'
    elif x in ['catalunya','espaa','spain']:
        return 'spa'
    elif x in ['nz','newzealand']:
        return 'nzl'
    elif x in ['australia','austria']:
        return 'aus'
    elif x in ['switzerland','lasuisse']:
        return 'che'
    else :
        return x

def author_cnt_map(x: int) -> int:

    x = int(x)
    if x <= 5:
        return 0
    elif x <= 10:
        return 1
    elif x < 100:
        return 2
    else:
        return 3

def publisher_cnt_map(x: int) -> int:

    x = int(x)
    if x <= 5:
        return 0
    elif x <= 10:
        return 1
    elif x < 100:
        return 2
    elif x < 300:
        return 3
    else:
        return 4

def remove_outlier_by_age( target_data : pd.DataFrame, target_age:int)->pd.DataFrame:
    
    if 'age' not in target_data.columns:
        raise Exception( '[delete_outlier_of_age] age column not in target_data')
    
    tmp_idx = target_data[target_data['age'] > target_age].index
    
    return target_data.drop(tmp_idx)

def process_location( target_data : pd.DataFrame, process_level:int )->pd.DataFrame:
    
    if 'location' not in target_data.columns:
        raise Exception( '[preprocess_location] location column not in target_data')

    level_map = { 1 : 'country' , 2 : 'state', 3:'city'}
    # location 특수 문자 제거
    basic_str = 'location_'
    
    # u['location_country'] = u['location'].apply(lambda x: x.split(',')[2] if len(x.split(','))==3 else x.split(',')[3] )
    # u['location_state'] = u['location'].apply(lambda x: x.split(',')[1] if len(x.split(','))==3 else ( x.split(',')[1] if x.split(',')[2]=='' else x.split(',')[2] ))
    # u['location_city'] = u['location'].apply(lambda x: x.split(',')[0])
    
    if process_level >= 1 : 
        cur_str = basic_str + level_map[1]
        # target_data[cur_str] = target_data['location'].apply(lambda x: x.split(',')[2])
        target_data[cur_str] = target_data['location'].apply(lambda x: x.split(',')[2] if len(x.split(','))==3 else x.split(',')[3] )
        target_data[cur_str] = target_data[cur_str].apply(country_map)
        target_data[cur_str] = target_data[cur_str].apply(lambda x: re.sub('[^0-9a-zA-Z]','',x).strip())

    if process_level >= 2 :
        cur_str = basic_str + level_map[2]
        # target_data[cur_str] = target_data['location'].apply(lambda x: x.split(',')[1])
        target_data['location_state'] = target_data['location'].apply(lambda x: x.split(',')[1] if len(x.split(','))==3 else ( x.split(',')[1] if x.split(',')[2]=='' else x.split(',')[2] ))
        target_data[cur_str] = target_data[cur_str].apply(lambda x: re.sub('[^0-9a-zA-Z]','',x).strip())
        
    
    if process_level >= 3 :
        cur_str = basic_str + level_map[3]
        target_data[cur_str] = target_data['location'].apply(lambda x: x.split(',')[0])
        target_data[cur_str] = target_data[cur_str].apply(lambda x: re.sub('[^0-9a-zA-Z]','',x).strip())
        
    country_unique = target_data['location_country'].unique().tolist()
    for country in country_unique:
        if target_data[target_data['location_country']==country].shape[0]< 5:
            target_data.loc[target_data['location_country']==country,"location_country"] = 'Remove_Point'
    target_data.drop(target_data[target_data['location_country']=='Remove_Point'].index,inplace=True)
    # target_data.drop(target_data[target_data['location_country']==''].index,inplace=True)
    
    target_data = target_data.drop(['location'], axis=1)
    return target_data

# boosting model test 를 위한 임시 함수 
def process_location_v2( target_data : pd.DataFrame, process_level:int )->pd.DataFrame:
    
    if 'location' not in target_data.columns:
        raise Exception( '[preprocess_location] location column not in target_data')

    level_map = { 1 : 'country' , 2 : 'state', 3:'city'}
    # location 특수 문자 제거
    basic_str = 'location_'
    
    # u['location_country'] = u['location'].apply(lambda x: x.split(',')[2] if len(x.split(','))==3 else x.split(',')[3] )
    # u['location_state'] = u['location'].apply(lambda x: x.split(',')[1] if len(x.split(','))==3 else ( x.split(',')[1] if x.split(',')[2]=='' else x.split(',')[2] ))
    # u['location_city'] = u['location'].apply(lambda x: x.split(',')[0])
    
    if process_level >= 1 : 
        cur_str = basic_str + level_map[1]
        # target_data[cur_str] = target_data['location'].apply(lambda x: x.split(',')[2])
        target_data[cur_str] = target_data['location'].apply(lambda x: x.split(',')[2] if len(x.split(','))==3 else x.split(',')[3] )
        target_data[cur_str] = target_data[cur_str].apply(country_map)
        target_data[cur_str] = target_data[cur_str].apply(lambda x: re.sub('[^0-9a-zA-Z]','',x).strip())

    if process_level >= 2 :
        cur_str = basic_str + level_map[2]
        # target_data[cur_str] = target_data['location'].apply(lambda x: x.split(',')[1])
        target_data['location_state'] = target_data['location'].apply(lambda x: x.split(',')[1] if len(x.split(','))==3 else ( x.split(',')[1] if x.split(',')[2]=='' else x.split(',')[2] ))
        target_data[cur_str] = target_data[cur_str].apply(lambda x: re.sub('[^0-9a-zA-Z]','',x).strip())
        
    
    if process_level >= 3 :
        cur_str = basic_str + level_map[3]
        target_data[cur_str] = target_data['location'].apply(lambda x: x.split(',')[0])
        target_data[cur_str] = target_data[cur_str].apply(lambda x: re.sub('[^0-9a-zA-Z]','',x).strip())
        
    country_unique = target_data['location_country'].unique().tolist()
    for country in country_unique:
        if target_data[target_data['location_country']==country].shape[0]< 5:
            target_data.loc[target_data['location_country']==country,"location_country"] = 'Remove_Point'
    target_data.drop(target_data[target_data['location_country']=='Remove_Point'].index,inplace=True)
    # target_data.drop(target_data[target_data['location_country']==''].index,inplace=True)
    
    # target_data = target_data.drop(['location'], axis=1)
    return target_data


def process_age( target_data : pd.DataFrame,  how : object = 'mean') -> pd.DataFrame:
    """
    target dataframe 에 age column 이 있을 경우, 
    column 의 결측치를  'how' parameter 에 전달된 값으로 채운다.
    - 현재는 'mean'만 적용 
    """

    if 'age' not in target_data.columns:
        print( '[process_age] age column not in target_data')
        return None
    
    if  'mean' == how :
        target_data['age'] = target_data['age'].fillna(int(target_data['age'].mean()))

    target_data['age'] = get_apply_map_series(target_data,'age', age_map)

    return target_data

# gu 단골 추가
def add_regular_custom( target_data:pd.DataFrame, col:str)->pd.DataFrame:  

    common = target_data.groupby([col, 'user_id'])[['rating']].count()
    common_count = common[common['rating']>2].groupby(col).count().sort_values('rating', ascending=False).rename(columns={'rating': col + '_common_cnt'}).reset_index()
    target_data = target_data.merge(common_count, on=col, how='left')
    target_data[col + '_common_cnt'].fillna(0, inplace=True)
    if col == 'book_author':
        target_data[col + '_common_cnt'] = get_apply_map_series(target_data,col + '_common_cnt',author_cnt_map)
    elif col == 'publisher':
        target_data[col + '_common_cnt'] = get_apply_map_series(target_data,col + '_common_cnt',publisher_cnt_map)
    return target_data