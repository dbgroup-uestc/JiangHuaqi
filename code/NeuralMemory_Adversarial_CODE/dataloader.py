import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
import pdb


def movielens(path, rating_thres = 1, seq_len = 1, split_ratio = (4, 2, 2, 2, 2)):
    ratings = pd.read_csv(path, names = ['user', 'item', 'rating', 'time'], header = 0,
                          dtype = {'user': np.int32, 'item': np.int32, 'rating': np.float64, 'time': np.int32})


    # unique_users = np.sort(ratings['user'].unique())
    # unique_items = np.sort(ratings['item'].unique())
    # unique_times = np.sort(ratings['time'].unique())
    # print(ratings.shape)
    # print(len(unique_users))
    # print(len(unique_items))
    # ratings = ratings[ratings['rating'] < 3]
    # print(ratings.shape)
    #
    # user_remap = {}
    # for i, user in enumerate(unique_users):
    #     user_remap[user] = i
    # ratings['user'] = ratings['user'].map(lambda x : user_remap[x])
    # item_remap = {}
    # for i, item in enumerate(unique_items):
    #     item_remap[item] = i
    # ratings['item'] = ratings['item'].map(lambda x : item_remap[x])


    ratings = ratings[ratings['rating'] >= rating_thres]
    user_cnt = ratings['user'].value_counts().to_dict()
    ratings = ratings[ratings['user'].map(lambda x: user_cnt[x] >= seq_len)]
    most_popular_items = ratings.groupby('item').size().sort_values(ascending = False)[:500].index.values.tolist()
    n_items = ratings['item'].max() + 1
    n_users = ratings['user'].max() + 1
    ratings = ratings.sort_values(by = ['time'])
    ratings_len = ratings.shape[0]
    train_len = int(ratings_len * split_ratio[0] / sum(split_ratio))
    test1_len = int(ratings_len * split_ratio[1] / sum(split_ratio))
    test2_len = int(ratings_len * split_ratio[2] / sum(split_ratio))
    test3_len = int(ratings_len * split_ratio[3] / sum(split_ratio))
    # test4_len = int(ratings_len * split_ratio[1] / sum(split_ratio))
    train_pd, test1_pd, test2_pd, test3_pd, test4_pd = np.split(ratings, [train_len, train_len + test1_len,
                                                                          train_len + test1_len + test2_len,
                                                                          train_len + test1_len + test2_len + test3_len])

    return train_pd, test1_pd, test2_pd, test3_pd, test4_pd, most_popular_items, n_users, n_items


if __name__ == '__main__':
    train_pd, test1_pd, test2_pd, test3_pd, test4_pd, most_popular_items, n_users, n_items = movielens('datasets/ml/ratings.csv')

