import pandas as pd

def make_item_2_idx_map( target_key_list:list, target_data:pd.DataFrame ) :
    
    idx_map_dict = {}

    for key in target_key_list : 
        dict_key = key + '2idx'
        cur_idx_map = {idx:cur_value for idx,cur_value in enumerate(target_data[key].unique())}
        idx_map_dict[dict_key] = cur_idx_map

    return idx_map_dict

def make_idx2_item_map( target_key_list:list, target_data:pd.DataFrame ) :

    item_map_dict = {}

    for key in target_key_list : 
        dict_key = 'idx2'+ key 
        cur_item_map = {cur_value:idx for idx,cur_value in enumerate(target_data[key].unique())}
        item_map_dict[dict_key] = cur_item_map

    return item_map_dict

def process_str_column( target_key_list:list, target_data:pd.DataFrame ) ->pd.DataFrame :
    """
    전달된 key list 가 object type 인 경우, 문자열 처리 해준다. 
    - 숫자, 영어, ',' 외 모든 문자 대치
    - 소문자 변경 
    """
    try:
        for key in target_key_list :
            if key not in target_data.columns:
                print('[WARN][remove_special_char_of_str] ', key, ' is not element of target_data')
                continue 

            target_data[key] = target_data[key].str.replace(r'[^0-9a-zA-Z:,]', '',regex = True) # 특수문자 제거
            target_data[key] = target_data[key].str.lower() # 소문자 변경 
        return target_data
    except Exception as e :
        print("[WARN][process_str_column] There is something wrong ",e)
        return None

def get_apply_map_series( target_data: pd.DataFrame, target_key:str, target_map:object ):

    if target_key not in target_data.columns:
        print("[WARN][get_apply_map_series] ", target_key, " is not in target_data")

    return target_data[target_key].apply(target_map)

def get_cnt_series_by_column( target_data:pd.DataFrame, target_key : str, orient_key:str)->pd.Series:
    
    if( target_key not in target_data):
        print('[WARN][get_cnt_series_by_column] ', target_key, ' is not element of target_data')
        return None 

    if( orient_key not in target_data):
        print('[WARN][get_cnt_series_by_column] The primary key is not element of target_data')
        return None 

    target_dict = target_data.groupby(target_key).count()[orient_key].sort_values()
    return target_data[target_key].map(target_dict)

# add_regular_custom_by_author의 일반화를 위함
# 20221031 사용하지 않음 
def add_regular_cnt_by_multi_column( target_data : pd.DataFrame, target_key_list : list , orient_key: str):
    """
    두 개의 컬럼을 기준으로 count 를 세어 2개 이상인 것의 개수를 센다.
    주요 컬럼을 target_key_list 의 첫 번째로 배치
    """

    new_column_name = second_oriend_col + "_common_cnt"
    first_orient_col, second_oriend_col = target_key_list  
    common = target_data.groupby([target_key_list])[[orient_key]].count()
    author_common = common[common[orient_key]>2].groupby(first_orient_col)\
                                                .count()\
                                                .sort_values(orient_key, ascending=False)\
                                                .rename(columns={orient_key: new_column_name})\
                                                .reset_index()
    target_data = target_data.merge(author_common, on=first_orient_col, how='left')
    target_data[new_column_name].fillna(0, inplace=True)
    
    return target_data