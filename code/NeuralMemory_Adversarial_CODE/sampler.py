import numpy as np
from dataloader import movielens
import pdb
from tqdm import tqdm


class WARPSampler():
    def __init__(self, train_pd, most_popular_items, n_items, batch_size = 64, n_candiidates = 10,
                 check_negative = True):
        """
        :param user_item_matrix: the user-item matrix for positive user-item pairs
        :param batch_size: number of samples to return
        :param n_negative: number of negative samples per user-positive-item pair
        :param result_queue: the output queue
        :return: None
        """
        self.train_pd = train_pd
        self.batch_size = batch_size
        self.n_items = n_items
        self.n_candiidates = n_candiidates
        self.check_negative = check_negative

        self.items_set = set(list(range(1, n_items)))
        self.user_to_items = dict()
        for t in self.train_pd.itertuples():
            self.user_to_items.setdefault(t.user, set())
            self.user_to_items[t.user].add(t.item)
        # self.items_set = set(list(range(1, n_items)))
        # self.user_negative_samples = dict()
        # for u in tqdm(self.user_to_items.keys(), desc = 'sampling user negative items for sampler'):
        #     self.user_negative_samples[u] = list(self.items_set - self.user_to_items[u])

    def next_batch(self):
        # self.train_pd = self.train_pd.sort_values(by = ['user', 'time'])
        for i in range(int(self.train_pd.shape[0] / self.batch_size)):
            user_positive_items_pairs = self.train_pd.iloc[i * self.batch_size: (i + 1) * self.batch_size][
                ['user', 'item']].values
            # negative_cands = np.zeros((self.batch_size, self.n_candiidates))
            # for j, u in enumerate(user_positive_items_pairs[:, 0]):
            #     negative_cands[j] = np.random.choice(self.user_negative_samples[u], self.n_candiidates)
            # negative_cands = negative_cands.astype(int)
            # yield user_positive_items_pairs, negative_cands
            negative_samples = np.random.randint(1, self.n_items, size = (self.batch_size, self.n_candiidates))
            if self.check_negative:
                for (j, k), neg in np.ndenumerate(negative_samples):
                    while neg in self.user_to_items[user_positive_items_pairs[j, 0]]:
                        negative_samples[j, k] = neg = np.random.randint(0, self.n_items)
            yield user_positive_items_pairs, negative_samples




if __name__ == '__main__':
    train_pd, test_pd, most_popular_items, n_items = movielens('datasets/ml/ml20m.csv')
    sampler = WARPSampler(train_pd, most_popular_items, n_items, batch_size = 10, n_candiidates = 200,
                          check_negative = True)
    for user_pos, neg_cands in sampler.next_batch():
        print(neg_cands)
    # for user_pos, neg in sampler.next_batch():
    #     pos_user = user_pos[:, 0]
