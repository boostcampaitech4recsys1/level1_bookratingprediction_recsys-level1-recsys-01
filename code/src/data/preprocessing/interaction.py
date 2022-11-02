import pandas as pd

from surprise import Dataset, Reader, accuracy, SVD, KNNBasic, CoClustering
from surprise.dataset import DatasetAutoFolds

from pandas.api.types import CategoricalDtype
from scipy import sparse

def combine_features(ratings, X_train, X_test):
    # make interactions
    size_uid = ratings["user_id"].unique()
    size_iid = ratings["isbn"].unique()

    ui_shape = (len(size_uid), len(size_iid))

    user_cat = CategoricalDtype(categories=sorted(size_uid), ordered=True)
    book_cat = CategoricalDtype(categories=sorted(size_iid), ordered=True)

    user_index = ratings["user_id"].astype(user_cat).cat.codes
    book_index = ratings["isbn"].astype(book_cat).cat.codes

    interactions = sparse.coo_matrix((ratings["rating"], (user_index,book_index)), shape=ui_shape)
    
    # ratings > train, test split
    uids, iids, data = shuffle_data(interactions)
    train_idx, test_idx = cutoff_by_user(uids)

    train_df = pd.DataFrame({'uid':uids[train_idx], 'iid':iids[train_idx], 'ratings':data[train_idx]})
    test_df = pd.DataFrame({'uid':uids[test_idx], 'iid':iids[test_idx], 'ratings':data[test_idx]})

    # train_set
    reader = Reader(rating_scale=(1, 10))
    fold = DatasetAutoFolds(df = train_df[['uid', 'iid', 'ratings']], reader = reader)
    trainset = fold.build_full_trainset()
    
    # SVD
    svd = SVD(trainset)
    svd_test = svd.test(list(zip(test_df['uid'], test_df['iid'], test_df['ratings'])))
    svd_pred = list(map(lambda x:x.est, svd_test))

    # co-clustering
    coclu = CoClustering(trainset)
    coclu_test = coclu.test(list(zip(test_df['uid'], test_df['iid'], test_df['ratings'])))
    coclu_pred = list(map(lambda x:round(x.est), coclu_test))

    # add column
    train_uir_tuple = list(zip(train_df['uid'], train_df['iid'], train_df['ratings']))

    X_train['svd_rating'] = list(map(lambda x:x.est, svd.test(train_uir_tuple)))
    X_test['svd_rating'] = svd_pred
    X_train['coclu_rating'] = list(map(lambda x:x.est, coclu.test(train_uir_tuple)))
    X_test['coclu_rating'] = coclu_pred

    return X_train, X_test


def shuffle_data(interactions:sparse.coo_matrix, random_state:int=42)->tuple:
    random_state = np.random.RandomState(seed=random_state)

    interactions = interactions.tocoo()

    uids, iids, data = (interactions.row, interactions.col, interactions.data)

    shuffle_indices = np.arange(len(uids))
    random_state.shuffle(shuffle_indices)

    uids = uids[shuffle_indices]
    iids = iids[shuffle_indices]
    data = data[shuffle_indices]

    return uids, iids, data


def cutoff_by_user(uids:list, test_percentage:float=0.2):
    cutoff = int((1.0 - test_percentage) * len(uids))
    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)
    return train_idx, test_idx
    

def SVD(trainset):
    params = {'n_factors':100,
            'n_epochs':20,
            'lr_all':0.005,
            'reg_all':0.02}

    svd = SVD(**params,random_state=42)
    svd.fit(trainset)

    return svd


def CoClusting(trainset):
    params = {'n_cltr_u':10,
            'n_cltr_i':10, 
            'n_epochs':20}

    coclu = CoClustering(**params, random_state=42, verbose=True)
    coclu.fit(trainset)

    return coclu


def check_sparsity(interactions:sparse.coo_matrix)->int:
    matrix_size = interactions.shape[0]*interactions.shape[1] # Number of possible interactions in the matrix
    num_purchases = len(interactions.nonzero()[0]) # Number of items interacted with
    sparsity = 100*(1 - (num_purchases/matrix_size))
    return round(sparsity,4)