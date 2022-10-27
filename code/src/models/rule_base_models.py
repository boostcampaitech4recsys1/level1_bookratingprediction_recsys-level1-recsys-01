import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _RuleBaseModel
from ._models import rmse, RMSELoss


class RuleBaseModel:

    def __init__(self, args, data):
        super().__init__()
        print(data.keys() )
        self.criterion = RMSELoss()
        self.data = data

    def train(self):
        print("not working")

    def apply_rule_base(self,cur_user_id, cur_book_author, cur_category_rank):

        cur_rating = 5.7

        author_dict = self.data['popular']['author'] 
        reader_dict = self.data['popular']['reader'] 
        category_dict = self.data['popular']['category'] 
        reader_with_category_dict = self.data['popular']['reader_with_category'] 

        if cur_user_id is not None :
            if cur_category_rank is not None:
                item = reader_with_category_dict[ (cur_user_id == reader_with_category_dict['user_id']) &(cur_category_rank == reader_with_category_dict['category_rank']) ]['rating']
                if(len(item) > 0):
                    cur_rating = item.item()
            else:
                item = reader_dict[(cur_user_id == reader_dict['user_id'])]['rating']
                if(len(item) > 0):
                    cur_rating = item.item()
        else:
            cur_rating = 5.7
        cur_rating = 5.7 if cur_rating < 1.0 else cur_rating
               
        return cur_rating 

    def predict_train(self):
        targets, predicts = self.data['y_valid'].tolist(), list()

        for idx in self.data['X_valid'].index:
            _cur_user_id =  self.data['X_valid'].loc[idx,'user_id']
            _cur_book_author =  self.data['X_valid'].loc[idx,'book_author']
            _cur_category_rank =  self.data['X_valid'].loc[idx,'category_rank']

            cur_predict = self.apply_rule_base(_cur_user_id, _cur_book_author, _cur_category_rank)
            predicts.append(cur_predict)

        return rmse(targets, predicts)


    def predict(self):
        predicts = list()

        for idx in self.data['test'].index:
            _cur_user_id = self.data['test'].loc[idx,'user_id']
            _cur_book_author = self.data['test'].loc[idx,'book_author']
            _cur_category_rank = self.data['test'].loc[idx,'category_rank']

            cur_predict = self.apply_rule_base(_cur_user_id, _cur_book_author, _cur_category_rank)
            predicts.append(cur_predict)

        print(predicts)
        return predicts