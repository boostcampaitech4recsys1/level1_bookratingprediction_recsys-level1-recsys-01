import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import sys 
import os

# _models 추가를 위한 상위 폴더 참조 추가 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from _models import rmse, RMSELoss

# optuna 의 object 에서 사용할 parameter 를 가져오는 함수 
def get_parameter( trial: Trial, model_name: str ):

    model_name_list = ['CatBoostRegressor','XGBRegressor', "LightGBMRegressor" ]
    
    if( model_name not in model_name_list ):
        print("[ERROR] this model is not provided. ", model_name)
        return {}

    if( model_name_list[0] == model_name ):
        return {
        "random_state":42,
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        "n_estimators":trial.suggest_int("n_estimators", 1000, 10000),
        "max_depth":trial.suggest_int("max_depth", 4, 16),
        'random_strength' :trial.suggest_int('random_strength', 0, 100),
        "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        }
    
    if(model_name_list[1] == model_name):
        return {
            "max_depth": trial.suggest_int("max_depth", 6, 10),
            "learning_rate": trial.suggest_uniform('learning_rate', 0.0001, 0.99),
            'n_estimators': trial.suggest_int("n_estimators", 400, 4000, 400),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e4), # L2 regularization
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e4), # L1 regularization
            'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e4),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.2, 0.9, 0.1),     
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 1e4),
        }

    if(model_name_list[2] == model_name):
        return {
             "objective": 'regression', # 회귀
             "eval_metric": "rmse",
             "verbosity": -1,
             "boosting_type": "gbdt",
             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
             "num_leaves": trial.suggest_int("num_leaves", 2, 256),
             "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
             "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
             "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    
    print("[WARN] this model is not implemented ", model_name )
    return {}

# optuna 의 study 에서 사용할 objective
def objective(trial: Trial, cur_data, cur_model, model_name,early_stopping_rounds = 35, verbose=100 ):
    
    X = cur_data['train'].drop('rating',axis=1)
    y = cur_data['train']['rating']
    
    param = {}
    score = None

    param = get_parameter(trial, model_name)
    
    print(param)
    # catboost 모델 cat_feature 지정을 위해 파라미터 추가 
    if( 'CatBoostRegressor'== model_name ):
        param['cat_features'] = cur_data['cat_features']

    cur_model = cur_model(**param)
    print(cur_model)
    fitted_model = cur_model.fit(X, y,  early_stopping_rounds=35, verbose=100)
    preds = fitted_model.predict(X)

    if( "Classifier" in model_name):
        preds = preds.squeeze(1)

    score = rmse( y, preds)

    return score

def do_optuna(cur_data, model, model_name, early_stopping_rounds = 35, verbose=100):
    # direction : score 값을 최대 또는 최소로 하는 방향으로 지정 
    study = optuna.create_study(direction='minimize',sampler=TPESampler())      
    # n_trials : 시도 횟수 (미 입력시 Key interrupt가 있을 때까지 무한 반복)
    study.optimize(lambda trial : objective(trial, cur_data,model ,model_name,early_stopping_rounds,verbose), n_trials=50)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))        

    try:
        # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
        optuna.visualization.plot_param_importances(study)      
        # 하이퍼파라미터 최적화 과정을 확인
        optuna.visualization.plot_optimization_history(study)
    except:
        pass 

    return study.best_params