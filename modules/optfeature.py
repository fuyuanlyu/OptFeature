import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.layer import FactorizationMachine, MultiLayerPerceptron, STE
import copy
import time
import modules.layer as layer

class NewFI(nn.Module):
    def __init__(self, args, device):
        super(NewFI, self).__init__()
        self.mode = args.fi_mode.lower()
        self.alpha_mode = args.alpha_mode.lower()
        self.stage = "search"

        self.rows = []
        self.cols = []
        for row in range(args.field):
            for col in range(row+1, args.field):
                self.rows.append(row)
                self.cols.append(col)
        self.rows = torch.tensor(self.rows, device=device)
        self.cols = torch.tensor(self.cols, device=device)

        self.device = device
        
        if self.mode == "value":
            self.fi_embedding_value = nn.Parameter(torch.zeros(args.feature, args.fi_dim))
            nn.init.xavier_uniform_(self.fi_embedding_value)
        elif self.mode == "field":
            self.fi_embedding_field = nn.Parameter(torch.zeros(args.field, args.fi_dim))
            nn.init.xavier_uniform_(self.fi_embedding_field)
            self.shallow_x = torch.from_numpy(np.arange(0, args.field)).to(self.device)
        elif self.mode == "hybrid":
            self.fi_embedding_value = nn.Parameter(torch.zeros(args.feature, args.fi_dim))
            nn.init.xavier_uniform_(self.fi_embedding_value)
            self.fi_embedding_field = nn.Parameter(torch.zeros(args.field, args.fi_dim))
            nn.init.xavier_uniform_(self.fi_embedding_field)
            self.shallow_x = torch.from_numpy(np.arange(0, args.field)).to(self.device)
            self.alpha = nn.Parameter(torch.zeros(int(args.field * (args.field - 1) / 2)))

        self.fi_dnn = nn.Linear(args.fi_dim, args.fi_mlp_dims[-1])
        self.fi_rank = nn.Parameter(torch.ones(args.fi_mlp_dims[-1]))
        self.sigma = STE.apply

    def get_FI_embedding(self, x, mode):
        if mode == "value":
            fi_embed = F.embedding(x, self.fi_embedding_value)
            fi_U = self.fi_dnn.forward(fi_embed)
        elif mode == "field":
            fi_embed = F.embedding(self.shallow_x.expand_as(x), self.fi_embedding_field)
            fi_U = self.fi_dnn.forward(fi_embed)
        p, q = self._unroll_embeddings(fi_U)
        fi = torch.sum(torch.mul(torch.matmul(p, torch.diag(self.fi_rank)), q), dim=2)
        return fi

    def generate_FI_vector(self, x):
        if self.mode == "value" or self.mode == "field":
            fi = self.get_FI_embedding(x, self.mode)
        elif self.mode == "hybrid":
            fi_v = self.get_FI_embedding(x, "value")
            fi_f = self.get_FI_embedding(x, "field")
            if self.stage == "search":
                fi = torch.sigmoid(self.alpha) * fi_v + (1 - torch.sigmoid(self.alpha)) * fi_f
            elif self.stage == "retrain":
                fi = (self.alpha.detach() > 0).float() * fi_v + (1 - (self.alpha.detach() > 0).float()) * fi_f
        return fi

    def generate_FI_vector_with_sigma(self, x):
        raw_fi = self.generate_FI_vector(x)
        fi = self.sigma(raw_fi)
        return fi

    def _unroll_embeddings(self, xv):
        batch_size = xv.shape[0]
        trans = torch.transpose(xv, 1, 2)
        gather_rows = torch.gather(trans, 2, self.rows.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans, 2, self.cols.expand(batch_size, trans.shape[1], self.cols.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        return p, q

class OptFeature(nn.Module):
    def __init__(self, args, device):
        super(OptFeature, self).__init__()
        self.args = args
        self.device = device
        self.embedding = nn.Parameter(torch.zeros(args.feature, args.dim))
        nn.init.xavier_uniform_(self.embedding)
        self.FI = NewFI(args, device)
        self.dnn_dim = args.field * args.dim + int(args.field * (args.field - 1)/2)
        self.inner = layer.InnerProduct(args.field)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, args.mlp_dims, output_layer=True, dropout=args.mlp_dropout, use_bn=args.mlp_bn)
    
    def generate_FI_vector(self, x):
        return self.FI.generate_FI_vector(x)
    
    def forward(self, x):
        x_embedding = F.embedding(x, self.embedding)
        x_dnn = x_embedding.view(-1, self.args.field * self.args.dim)
        x_fi = self.FI.generate_FI_vector_with_sigma(x)
        x_product = self.inner.forward(x_embedding)
        x_product = torch.mul(x_fi, x_product)
        x_input = torch.cat((x_dnn, x_product), 1)
        logit = self.dnn(x_input)
        return logit
