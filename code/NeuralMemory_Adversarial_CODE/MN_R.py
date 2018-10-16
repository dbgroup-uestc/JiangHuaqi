import model_R
import torch
from torch.autograd import Variable
import torch.optim as optim

from sampler import WARPSampler
from dataloader import movielens
import numpy as np
from tqdm import tqdm
import pdb
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    dim = 64
    memory_size = 128
    # 1000
    BATCH_SIZE = 1000
    # 200
    margin = 4
    use_rank_weight = True
    lr = 0.2
    user_negs_n = 5000
    n_negative = 10
    topk = 5
    train1_pd, test1_pd, test2_pd, test3_pd, test4_pd, most_popular_items, n_users, n_items = movielens('datasets/ml/ratings.csv')
    n_users = int(n_users)
    n_items = int(n_items)
    network = model_R.KVMRN(dim = dim, n_users = n_users, n_items = n_items, memory_size = memory_size)
    network = network.cuda()
    optimizer = optim.Adagrad(network.parameters(), lr = lr)


    # valid_users = valid_pd['user'].sample(1000).values

    test_pds = [test1_pd, test2_pd, test3_pd, test4_pd]
    # test_pds = [test1_pd]
    train_pd = train1_pd
    previous_test_pd = train1_pd

    for test_part, test_pd in enumerate(test_pds):
        train_users = train_pd['user'].values
        train_items = train_pd['item'].values
        all_users_in_train = set(list(train_users))
        all_items_in_train = set(list(train_items))
        user_to_train_set = dict()
        user_to_test_set = dict()
        for t in train_pd.itertuples():
            user_to_train_set.setdefault(t.user, set())
            user_to_train_set[t.user].add(t.item)
        for t in test_pd.itertuples():
            user_to_test_set.setdefault(t.user, set())
            user_to_test_set[t.user].add(t.item)
        sampler = WARPSampler(previous_test_pd, most_popular_items, n_items, batch_size = BATCH_SIZE,
                              n_candiidates = n_negative,
                              check_negative = True)
        epoch = 0
        while epoch < 10:
            epoch += 1
            for user_pos, neg_items in sampler.next_batch():
                optimizer.zero_grad()
                pos_users = user_pos[:, 0].astype(int)
                pos_items = user_pos[:, 1].astype(int)
                pos_users = Variable(torch.from_numpy(pos_users)).cuda()
                pos_items = Variable(torch.from_numpy(pos_items)).cuda()
                neg_items = Variable(torch.from_numpy(neg_items)).cuda()
                loss = network(pos_users, pos_items, neg_items, use_rank_weight, margin)
                loss.backward()
                optimizer.step()


        user_negative_samples = dict()
        items_set = set(list(range(1, n_items)))
        for u in tqdm(user_to_train_set.keys(), desc = 'sampling user negative items'):
            user_negative_samples[u] = np.random.choice(list(items_set - user_to_train_set[u]), user_negs_n)
        accs = []
        # all_items_embeddings = network.item_embeddings.weight
        for test_u in tqdm(list(user_to_test_set.keys()), desc = 'testing'):

            if test_u not in all_users_in_train:
                continue
            users_v = Variable(torch.from_numpy(np.array([test_u], dtype = int))).cuda()
            # [1, D]
            abst_prefers_embeds = network.abs_embed(users_v)
            hit = 0
            tot = 0
            for test_v in user_to_test_set[test_u]:
                if test_v not in all_items_in_train:
                    continue
                candidate_items = np.append(user_negative_samples[test_u], test_v)
                # [N, D]
                candidate_items_embeddings = network.item_embeddings(
                    Variable(torch.from_numpy(candidate_items)).cuda())
                item_scores = torch.sum((candidate_items_embeddings - abst_prefers_embeds) ** 2, dim = 1)
                # item_scores = item_scores.cpu().data.numpy()
                # user_tops = np.argpartition(item_scores, -topk)[-topk:]
                _, user_tops = torch.topk(item_scores, k = topk, largest = False)
                user_tops = user_tops.cpu().data.numpy()
                tot += 1
                if user_negs_n in user_tops:
                    hit += 1
            if tot > 0:
                accs.append(float(hit) / tot)
        print('Final accuracy@{} on test {} : {}'.format(topk, np.mean(accs), test_part + 1))
        previous_test_pd = test_pd
        train_pd = pd.concat([train_pd, test_pd])
