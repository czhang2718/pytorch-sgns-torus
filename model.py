# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch import LongTensor as LT
from torch import FloatTensor as FT


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, 
        vocab_size: int = 20000, 
        embedding_size: int = 300, 
        padding_idx: int = 0, 
        torus: bool = False,
    ):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        init_min = -0.5 / self.embedding_size if not torus else -1
        init_max = 0.5 / self.embedding_size if not torus else 1
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(init_min, init_max)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(init_min, init_max)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)



class SGNS(nn.Module):

    def __init__(self, 
        embedding: Bundler, 
        vocab_size: int = 20000, 
        n_negs: int = 20, 
        weights: np.ndarray | None = None,
        scale: float = 100,
        torus: bool = False
    ):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = weights
        self.h = self._batch_dot_product
        if torus:
            self.h = self._sum_cosine_diff
            self.coord_weights = nn.Parameter(t.ones(self.embedding.embedding_size)/t.sqrt(t.tensor(self.embedding.embedding_size, dtype=t.float32)))
            self.coord_weights.requires_grad = True

        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def _sum_cosine_diff(
        self,
        x: t.Tensor,  # [B, C1, E]
        y: t.Tensor  # [B, C2, E], C2=1
    ) -> t.Tensor:
        assert x.dim() == 3 and y.dim() == 3
        return (self.coord_weights * t.cos(t.pi*(x - y))).sum(dim=-1)

    def _batch_dot_product(
        self,
        x: t.Tensor, # [B, C1, E]
        y: t.Tensor # [B, C2, E], C2=1
    ) -> t.Tensor:
        assert x.dim() == 3 and y.dim() == 3
        return t.bmm(x, y.transpose(1, 2))


    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(1) # in [B, 1, E]
        ovectors = self.embedding.forward_o(owords) # out [B, C, E]
        nvectors = self.embedding.forward_o(nwords) # neg [B, C*N, E]

        o_inner_product = self.h(ovectors, ivectors)
        n_inner_product = self.h(nvectors, ivectors)
        
        # Debug: check signs and magnitudes
        # print(f"o_inner_product: mean={o_inner_product.mean().item():.4f} (should be high)")
        # print(f"n_inner_product: mean={n_inner_product.mean().item():.4f} (should be low)")
        oloss = F.logsigmoid(o_inner_product.squeeze()).mean(1)
        nloss = F.logsigmoid(-n_inner_product.squeeze()).view(-1, context_size, self.n_negs).sum(2).mean(1)

        # print("oloss", oloss, "nloss", nloss)
        return -(oloss + nloss).mean()
