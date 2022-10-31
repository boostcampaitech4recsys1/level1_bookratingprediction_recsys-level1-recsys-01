import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# from ._models import CatBoosting
from ._models import rmse, RMSELoss
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler

def objective(trial: Trial, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    param = {}
    param['learning_rate'] = trial.suggest_discrete_uniform("learning_rate", 0.001, 0.3, 0.001)
    param['depth'] = trial.suggest_int('depth', 9, 15)
    param['l2_leaf_reg'] = trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5)
    param['min_child_samples'] = trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32])
    param['grow_policy'] = 'Depthwise'
    param['iterations'] = 10000
    param['use_best_model'] = True
    param['eval_metric'] = 'RMSE'
    param['od_type'] = 'iter'
    param['od_wait'] = 20
    param['random_state'] = 42
    param['logging_level'] = 'Silent'
    param['bootstrap_type'] = 'Bernoulli'
    # param['use_model_best'] = true
    model = CatBoostRegressor(**param)
    catboost_model = model.fit(X_train.copy(), y_train.copy(), eval_set=[(X_test.copy(), y_test.copy())], early_stopping_rounds=35,verbose=100)

    ## RMSE으로 Loss 계산
    score = rmse( y_test,catboost_model.predict(X_test))
    print(score)

    return score

class CatBoostingModel:
    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        self.catmodels = {}
        self.data = data

        # # direction : score 값을 최대 또는 최소로 하는 방향으로 지정 
        # study = optuna.create_study(direction='minimize',sampler=TPESampler())      
        # # n_trials : 시도 횟수 (미 입력시 Key interrupt가 있을 때까지 무한 반복)
        # study.optimize(lambda trial : objective(trial, self.data['train'].drop('rating',axis=1), self.data['train']['rating']), n_trials=50)
        # print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))        
        
        # self.best_params = study.best_params
        # try:
        #     # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
        #     optuna.visualization.plot_param_importances(study)      
        #     # 하이퍼파라미터 최적화 과정을 확인
        #     optuna.visualization.plot_optimization_history(study)
        # except:
        #     pass    

    def train(self):
        print(self.data['folds'])
        for fold in range(8):
            print(f'===================================={fold+1}============================================')
            train_idx, valid_idx = self.data['folds'][fold]
            X_train = self.data['train'].drop(['rating'],axis=1).iloc[train_idx].values 
            X_valid = self.data['train'].drop(['rating'],axis=1).iloc[valid_idx].values
            y_train = self.data['train']['rating'][train_idx].values
            y_valid = self.data['train']['rating'][valid_idx].values

            cat = CatBoostRegressor( learning_rate= 0.048, 
                                    depth= 15,
                                    l2_leaf_reg= 1.5,
                                    min_child_samples= 1,
                                    eval_metric = 'RMSE',
                                    bootstrap_type = 'Bernoulli',
                                    iterations = 10000,
                                    grow_policy = 'Depthwise',
                                    use_best_model = True,
                                    od_type = 'Iter',
                                    od_wait = 20,
                                    random_state = 42,
                                    )

            # cat = CatBoostRegressor( **self.best_params,
            #                         random_state = 42
            #                         )
            cat.fit(X_train,y_train, eval_set=(X_valid, y_valid))
            self.catmodels[fold] = cat 
            print(f'================================================================================\n\n')
            #print('epoch:', epoch, 'validation: rmse:', rmse_score)
        print( **self.best_params)
        
        return 


    def predict(self):
        preds =np.zeros(self.data['test'].shape[0])  
        for fold in range(8):
            a = self.catmodels[fold].predict(self.data['test']).transpose()
            preds += self.catmodels[fold].predict(self.data['test'])
        print(preds)
        return (preds/8).transpose()