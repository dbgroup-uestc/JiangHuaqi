import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb


class Dis(nn.Module):
    def __init__(self, dim, n_users, n_items, memory_size = 64):
        super(Dis, self).__init__()
        self.memory_size = memory_size
        self.dim = dim
        self.n_users = n_users
        self.n_items = n_items
        self.key_matrix = Parameter(torch.Tensor(memory_size, dim))
        self.value_matrix = Parameter(torch.Tensor(memory_size, dim))
        self.user_embeddings = nn.Embedding(n_users, dim, max_norm = 1)
        self.item_embeddings = nn.Embedding(n_items, dim, max_norm = 1)

        self.e_linear = nn.Linear(dim, dim)
        self.a_linear = nn.Linear(dim, dim)
        self.key_matrix.data.normal_(0, 1)
        self.value_matrix.data.normal_(0, 1)

    def forward(self, users, items, neg_items, use_rank_weight, margin):
        # [batch_size]
        # embed = [batch_size, dim}
        users_embed = self.user_embeddings(users)
        items_embed = self.item_embeddings(items)
        neg_items_embed = self.item_embeddings(neg_items)
        # attentionW

        ex_users_embed = torch.unsqueeze(users_embed, 1)
        attention = torch.sum((ex_users_embed - self.key_matrix) ** 2, dim = 2)
        correlation_weight = F.softmax(-attention, dim = 1)
        # read

        # [1, memory size, dim]
        ex_value_matrx = torch.unsqueeze(self.value_matrix, dim = 0)
        # [batch size, memory size, 1]
        ex_correlation_weight = torch.unsqueeze(correlation_weight, dim = 2)
        abst_items_embed = torch.sum(ex_value_matrx * ex_correlation_weight, dim = 1)

        # write
        
        # [batch size, dim]
        erase_vector = self.e_linear(items_embed)
        erase_signal = F.sigmoid(erase_vector)
        add_vector = self.a_linear(items_embed)
        add_signal = F.tanh(add_vector)
        # [batch size, 1, dim]
        ex_erase_signal = torch.unsqueeze(erase_signal, 1)
        # w_t(i) * e_t
        # # [batch size, 1, dim]
        # [batch size, memory size, 1]
        erase_mul = torch.mean(ex_erase_signal * ex_correlation_weight, dim = 0)
        erase = self.value_matrix * (1 - erase_mul)
        
        add_reshaped = add_signal.view(-1, 1, self.dim)
        add_mul = torch.mean(add_reshaped * ex_correlation_weight, dim = 0)
        self.value_matrix.data = (erase + add_mul).data

        # pos_distances = torch.sum((users_embed - items_embed) ** 2, dim = 1)
        # distance_to_neg_items = torch.sum((torch.unsqueeze(users_embed, 1) - neg_items_embed) ** 2, dim = 2)
        # closest_negative_item_distances = torch.min(distance_to_neg_items, dim = 1)[0]


        pos_abst_distances = torch.sum((abst_items_embed - items_embed) ** 2, dim = 1)
        abst_distance_to_neg_items = torch.sum((torch.unsqueeze(abst_items_embed, 1) - neg_items_embed) ** 2, dim = 2)
        closest_abst_negative_item_distances = torch.min(abst_distance_to_neg_items, dim = 1)[0]
        loss_per_pair = torch.clamp(pos_abst_distances - closest_abst_negative_item_distances + margin, min = 0)
        # loss_per_pair = torch.clamp(
        #     pos_abst_distances - closest_abst_negative_item_distances + pos_distances - closest_negative_item_distances + margin,
        #     min = 0)
        if use_rank_weight:
            # indicator matrix for impostors (N x W)
            impostors = (torch.unsqueeze(pos_abst_distances, -1) - abst_distance_to_neg_items + margin) > 0
            # impostors = (torch.unsqueeze(pos_abst_distances, -1) - abst_distance_to_neg_items + torch.unsqueeze(pos_distances, -1) - distance_to_neg_items + margin) > 0
            rank = torch.mean(impostors.float(), dim = 1) * self.n_users
            loss_per_pair *= torch.log(rank + 1)
        loss = torch.mean(loss_per_pair)
        return loss

    def get_D_D(self, users, neg_item):
        # [batch_size]
        # embed = [batch_size, dim}
        users_embed = self.user_embeddings(users)
        neg_item_embed = self.item_embeddings(neg_item)
        neg_item_embed = torch.squeeze(neg_item_embed, 1)

        # attentionW

        ex_users_embed = torch.unsqueeze(users_embed, 1)
        attention = torch.sum((ex_users_embed - self.key_matrix) ** 2, dim = 2)
        correlation_weight = F.softmax(-attention, dim = 1)
        # read

        # [1, memory size, dim]
        ex_value_matrx = torch.unsqueeze(self.value_matrix, dim = 0)
        # [batch size, memory size, 1]
        ex_correlation_weight = torch.unsqueeze(correlation_weight, dim = 2)
        abst_item_embed = torch.sum(ex_value_matrx * ex_correlation_weight, dim = 1)
        D_D = torch.sum((abst_item_embed - neg_item_embed) ** 2, dim = 1)

        return D_D



    def abs_embed(self, users):
        # [N, D]
        users_embed = self.user_embeddings(users)
        # attentionW
        #
        ex_users_embed = torch.unsqueeze(users_embed, 1)
        attention = torch.sum((ex_users_embed - self.key_matrix) ** 2, dim = 2)
        correlation_weight = F.softmax(-attention, dim = 1)
        # read

        # [1, memory size, dim]
        ex_value_matrx = torch.unsqueeze(self.value_matrix, dim = 0)
        # [batch size, memory size, 1]
        ex_correlation_weight = torch.unsqueeze(correlation_weight, dim = 2)
        read_content = torch.sum(ex_value_matrx * ex_correlation_weight, dim = 1)
        return read_content


class Gen(nn.Module):
    def __init__(self, dim, n_users, n_items):
        super(Gen, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n1 = 128
        self.n2 = 32


        self.C = nn.Embedding(n_users, dim, max_norm = 1)
        self.D = nn.Embedding(n_items, dim, max_norm = 1)
        self.l1 = nn.Linear(dim, 128)
        self.l2 = nn.Linear(dim, 128)


    def forward(self, user, pos_items, neg_cands, n_neg):
        user_vector = self.C(user)
        user_vector = self.l1(user_vector)

        # items_vector = self.D(item)
        neg_cands_vector = self.D(neg_cands)
        neg_cands_vector = self.l2(neg_cands_vector)
        pos_items_vector = self.D(pos_items)
        pos_items_vector = self.l2(pos_items_vector)
        # pos_distances = torch.exp(-torch.sum((user_vector - items_vector) ** 2))
        D_G_p = torch.exp(-torch.sum((user_vector - pos_items_vector) ** 2, dim = 1))
        user_vector = torch.unsqueeze(user_vector, 1)
        D_G_n = torch.exp(-torch.sum((user_vector - neg_cands_vector) ** 2, dim = 2))
        probs = D_G_n / torch.sum(D_G_n, dim = 1, keepdim = True)
        # shape (batchsize, n_neg]
        # m = Categorical(probs)
        # neg_index = m.sample()
        # negs_index = m.sample_n(n_neg).t()
        # neg_index = torch.unsqueeze(neg_index, dim = 1)
        neg_index = torch.multinomial(probs, 1, False)
        negs_index = torch.multinomial(probs, n_neg, False)
        return D_G_p, D_G_n, probs, neg_index, negs_index
