from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ImageQueryGraphConvolution
from layers import TextQueryGraphConvolution
import numpy as np


class RBFKernel(nn.Module):
    def __init__(self, sigma=1.0):
        super(RBFKernel, self).__init__()
        self.sigma = sigma  # 核的宽度参数（超参数）

    def forward(self, X, Y=None):
        """
        计算 RBF 核相似度矩阵
        :param X: (B, N, D) 其中B是batch size，N是样本数量，D是嵌入维度
        :param Y: (B, M, D)，可选，默认与 X 相同
        :return: (B, N, M) 的协方差矩阵
        """
        if Y is None:
            Y = X  # 如果没有提供 Y，默认 Y 与 X 相同

        # 计算欧几里得距离
        dist = torch.sum((X.unsqueeze(2) - Y.unsqueeze(1)) ** 2, dim=-1)  # (B, N, M)

        # 计算 RBF 核，相当于：exp(-d^2 / (2 * sigma^2))
        kernel_matrix = torch.exp(-dist / (2 * self.sigma ** 2))

        return kernel_matrix

# 定义高斯过程嵌入头类
class GaussianProcessEmbeddingHead(nn.Module):
    def __init__(self, in_dim, embed_dim, sigma=1.0):
        super(GaussianProcessEmbeddingHead, self).__init__()
        self.fc_mu = nn.Linear(in_dim, embed_dim)  # 均值
        self.fc_logvar = nn.Linear(in_dim, embed_dim)  # 对数方差

        # 高斯过程的协方差函数
        self.kernel = RBFKernel(sigma)  # 例如RBF核，用于计算协方差矩阵

    def forward(self, x):
        # 计算均值和对数方差
        mu = self.fc_mu(x)  # 均值
        logvar = self.fc_logvar(x)  # 对数方差
        sigma = torch.exp(0.5 * logvar)  # 计算标准差

        covariance_matrix = self.kernel(x)
        cov_diag = covariance_matrix.diagonal(offset=0, dim1=1, dim2=2)
        sigma_adjusted = sigma * cov_diag.unsqueeze(-1)

        return mu, sigma_adjusted

def l1norm(X, dim, eps=1e-8):

    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def inter_relations(K, Q, xlambda):

    batch_size, queryL = Q.size(0), Q.size(1)
    batch_size, sourceL = K.size(0), K.size(1)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    queryT = torch.transpose(Q, 1, 2)

    attn = torch.bmm(K, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn


def intra_relations(K, Q, xlambda):

    batch_size, KL = K.size(0), K.size(1)
    K = torch.transpose(K, 1, 2).contiguous()
    attn = torch.bmm(Q, K)

    attn = attn.view(batch_size * KL, KL)
    #attn = nn.Softmax(dim=1)(attn)
    # attn = l2norm(attn, -1)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    attn = attn.view(batch_size, KL, KL)
    return attn


class VisualGraph(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 dropout,
                 n_kernels=8):


        super(VisualGraph, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim

        # graph convolution layers
        self.graph_convolution_1 = \
            ImageQueryGraphConvolution(feat_dim, hid_dim, n_kernels, 2)

        self.gauss_head = GaussianProcessEmbeddingHead(hid_dim, hid_dim)
        # # output classifier
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))

    def node_level_matching(self, tnodes, vnodes, n_block, xlambda):
        # Node-level matching: find relevant nodes from another modality
        inter_relation = inter_relations(tnodes, vnodes, xlambda)

        # Compute sim with weighted context
        # (batch, n_word, n_region)
        attnT = torch.transpose(inter_relation, 1, 2)
        contextT = torch.transpose(tnodes, 1, 2)  # (batch, dim, n_word)
        weightedContext = torch.bmm(contextT, attnT)  # (batch, dim, n_region)
        weightedContextT = torch.transpose(
            weightedContext, 1, 2)  # (batch, n_region, dims)

        # Multi-block similarity
        # (batch, n_region, num_block, dims/num_block)
        qry_set = torch.split(vnodes, n_block, dim=2)
        ctx_set = torch.split(weightedContextT, n_block, dim=2)

        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)

        # (batch, n_region, num_block)
        vnode_mvector = cosine_similarity(
            qry_set, ctx_set, dim=-1)

        return vnode_mvector

    def structure_level_matching(self, vnode_mvector, pseudo_coord):
        # (batch, n_region, n_region, num_block)
        batch, n_region = vnode_mvector.size(0), vnode_mvector.size(1)
        neighbor_image = vnode_mvector.unsqueeze(
            2).repeat(1, 1, n_region, 1)

        # Propagate matching vector to neighbors to infer phrase correspondence
        hidden_graph = self.graph_convolution_1(neighbor_image, pseudo_coord)
        mu, sigma = self.gauss_head(hidden_graph)

        hidden_graph = hidden_graph.view(batch * n_region, -1)

        # Jointly infer matching score
        sim = self.out_2(self.out_1(hidden_graph).tanh())
        sim = sim.view(batch, -1).mean(dim=1, keepdim=True)

        return sim, mu, sigma

    def forward(self, images, captions, bbox, cap_lens, opt):
        similarities = []  # (n_image, n_caption)
        mu_mix_all = []  # [(B, R, E), ...]  per-caption
        sigma_mix_all = []

        n_block = opt.embed_size // opt.num_block
        n_image, n_caption = images.size(0), captions.size(0)

        bb_size = (bbox[:, :, 2:] - bbox[:, :, :2])
        bb_centre = bbox[:, :, :2] + 0.5 * bb_size

        pseudo_coord = self._compute_pseudo(bb_centre).cuda()
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            # --> compute similarity between query region and context word
            # --> (batch, n_region, n_word)
            vnode_mvector = self.node_level_matching(
                cap_i_expand, images, n_block, opt.lambda_softmax)
            v2t_similarity , mu_i, sigma_i = self.structure_level_matching(
                vnode_mvector, pseudo_coord)

            similarities.append(v2t_similarity)
            mu_mix_all.append(mu_i)          # 直接把 R 个分量留着
            sigma_mix_all.append(sigma_i)

        similarities = torch.cat(similarities, 1)  # (B, n_caption)
        v_m = torch.stack(mu_mix_all, dim=1)  # (B, n_caption, R, E)
        v_s = torch.stack(sigma_mix_all, dim=1)

        weights = F.softmax(similarities, dim=1)
        w = weights.unsqueeze(-1).unsqueeze(-1)
        mu_region = (w * v_m).sum(dim=1)
        var_region = (w * (v_s**2 + v_m**2)).sum(dim=1) - mu_region**2
        sigma_region = torch.sqrt(var_region.clamp(min=1e-8))

        return similarities,  mu_region, sigma_region

    def _compute_pseudo(self, bb_centre):

        K = bb_centre.size(1)

        # Compute cartesian coordinates (batch_size, K, K, 2)
        pseudo_coord = bb_centre.view(-1, K, 1, 2) - \
            bb_centre.view(-1, 1, K, 2)

        # Conver to polar coordinates
        rho = torch.sqrt(
            pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)
        theta = torch.atan2(
            pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
        pseudo_coord = torch.cat(
            (torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)

        return pseudo_coord


class TextualGraph(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 dropout,
                 n_kernels=8):


        super(TextualGraph, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim

        self.graph_convolution_3 = \
            TextQueryGraphConvolution(feat_dim, hid_dim, n_kernels, 2)

        self.gauss_head = GaussianProcessEmbeddingHead(hid_dim, hid_dim)  # ← 新增
        # # output classifier
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))

    def build_sparse_graph(self, dep, lens):
        adj = np.zeros((lens, lens), dtype=np.int)
        for i, pair in enumerate(dep):
            if i == 0 or pair[0] >= lens or pair[1] >= lens:
                continue
            adj[pair[0], pair[1]] = 1
            adj[pair[1], pair[0]] = 1
        adj = adj + np.eye(lens)
        return torch.from_numpy(adj).cuda().float()

    def node_level_matching(self, vnodes, tnodes, n_block, xlambda):

        inter_relation = inter_relations(vnodes, tnodes, xlambda)

        # Compute sim with weighted context
        # (batch, n_region, n_word)
        attnT = torch.transpose(inter_relation, 1, 2)
        contextT = torch.transpose(vnodes, 1, 2)  # (batch, dim, n_region)
        weightedContext = torch.bmm(contextT, attnT)  # (batch, dim, n_word)
        weightedContextT = torch.transpose(
            weightedContext, 1, 2)  # (batch, n_word, dims)

        # Multi-block similarity
        # (batch, n_word, num_block, dims/num_block)
        qry_set = torch.split(tnodes, n_block, dim=2)
        ctx_set = torch.split(weightedContextT, n_block, dim=2)

        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)

        tnode_mvector = cosine_similarity(
            qry_set, ctx_set, dim=-1)  # (batch, n_word, num_block)
        return tnode_mvector

    def structure_level_matching(self, tnode_mvector, intra_relation, depends, opt):
        # (batch, n_word, 1, num_block)
        tnode_mvector = tnode_mvector.unsqueeze(2)
        batch, n_word = tnode_mvector.size(0), tnode_mvector.size(1)

        adj_mtx = self.build_sparse_graph(depends, n_word)
        adj_mtx = adj_mtx.view(n_word, n_word).unsqueeze(0).unsqueeze(-1)
        # (batch, n_word, n_word, num_block)
        neighbor_nodes = adj_mtx * tnode_mvector
        # (batch, n_word, n_word, 1)
        neighbor_weights = l2norm(adj_mtx * intra_relation, dim=2)
        neighbor_weights = neighbor_weights.repeat(batch, 1, 1, 1)

        hidden_graph = self.graph_convolution_3(neighbor_nodes, neighbor_weights)

        mu, sigma = self.gauss_head(hidden_graph)
        hidden_graph = hidden_graph.view(batch * n_word, -1)

        sim = self.out_2(self.out_1(hidden_graph).tanh())
        sim = sim.view(batch, -1).mean(dim=1, keepdim=True)
        return sim,mu, sigma

    def forward(self, images, captions, depends, cap_lens, opt):
        n_image = images.size(0)
        n_caption = captions.size(0)
        similarities, mu_mix_all, sigma_mix_all = [], [], []
        n_block = opt.embed_size // opt.num_block
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()

            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            # --> compute similarity between query region and context word
            # --> (1, n_word, n_word, 1)
            words_sim = intra_relations(
                cap_i, cap_i, opt.lambda_softmax).unsqueeze(-1)
            nodes_sim = self.node_level_matching(
                images, cap_i_expand, n_block, opt.lambda_softmax)

            phrase_sim, mu_i, sigma_i = self.structure_level_matching(
                nodes_sim, words_sim, depends[i], opt)

            similarities.append(phrase_sim)
            mu_mix_all.append(mu_i)           # (B,W,E)
            sigma_mix_all.append(sigma_i)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        t_m = torch.stack(mu_mix_all,    1)       # (B, n_caption, W, E)
        t_s = torch.stack(sigma_mix_all, 1)

        weights = F.softmax(similarities, dim=1)  # (B, C)
        w = weights.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        mu_word = (w * t_m).sum(dim=1)
        var_word = (w * (t_s**2 + t_m**2)).sum(dim=1) - mu_word**2
        sigma_word = torch.sqrt(var_word.clamp(min=1e-8))  # (B, W, E)

        return similarities, mu_word, sigma_word
