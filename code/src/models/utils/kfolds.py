
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import sys 
import os 

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from _models import rmse, RMSELoss

class KfoldWrapper :

    def __init__(self,args, wrapper_model_class , kfold_num):

        self.kfold_num = kfold_num
        self.my_class = wrapper_model_class
        self.model_name = args.MODEL
        self.cur_model_collect = {}

        skf = StratifiedKFold(n_splits = kfold_num, shuffle=True, random_state=42)
        folds = []

        for train_idx, valid_idx in skf.split( wrapper_model_class.data['train'].drop(['rating'], axis=1), 
                                               wrapper_model_class.data['train']['rating']):
            folds.append((train_idx,valid_idx))

        self.folds = folds 

    def train(self):
        for fold in range( self.kfold_num  ):
            print(f'==================================== {fold+1} ============================================')
            train_idx, valid_idx = self.folds[fold]
            X_train = self.my_class.data['train'].drop(['rating'],axis=1).iloc[train_idx] 
            X_valid = self.my_class.data['train'].drop(['rating'],axis=1).iloc[valid_idx]
            y_train = self.my_class.data['train']['rating'][train_idx]
            y_valid = self.my_class.data['train']['rating'][valid_idx]

            target_model = CatBoostRegressor

            if( "LightGBMRegressor" == self.model_name  ):
                target_model = LGBMRegressor
            elif( "XGBRegressor" == self.model_name  ): 
                target_model = XGBRegressor
            elif( "CatBoostRegressor"== self.model_name ): 
                target_model = CatBoostRegressor

            cur_predict_model = target_model( **self.my_class.best_params)

            if( "CatBoostRegressor"== self.model_name ): 
                cur_predict_model.fit(X_train,y_train, eval_set=(X_valid, y_valid))
            else:
                cur_predict_model.fit(X_train,y_train)

            preds = cur_predict_model.predict(X_valid)
            print(f'{fold} : RMSE : ', rmse(y_valid, preds))
            self.cur_model_collect[fold] = cur_predict_model 
            print(f'================================================================================\n\n')

    def predict(self):
        preds =np.zeros(self.my_class.data['test'].shape[0])  
        for fold in range(self.kfold_num):
            simple_pred = self.cur_model_collect[fold].predict(self.my_class.data['test'])
            preds += simple_pred.squeeze(1) if 'Classifier' in self.model_name else simple_pred
        return ( preds / self.kfold_num ).transpose()
