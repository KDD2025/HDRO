
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from ...basic.layers import EmbeddingLayer,MLP,EmbeddingLayerHDRO
from ...basic.activation import activation_layer


class MLP_adap_7_layer_2_adp(nn.Module):
    # 7 layers MLP with 2 adapter cells
    def __init__(self, features, domain_num, fcn_dims, hyper_dims, k ):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()

        # backbone network architecture
        for d in range(domain_num):
            domain_specific = nn.ModuleList()
            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], self.fcn_dim[3]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[3]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[3], self.fcn_dim[4]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[4]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[4], self.fcn_dim[5]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[5]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[5], self.fcn_dim[6]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[6]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[6], self.fcn_dim[7]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[7]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[7], 1))

            self.layer_list.append(domain_specific)

        # instance representation matrix initiation
        self.k = k
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        # u,v matrix initiation
        self.u.append(Parameter(torch.ones((self.fcn_dim[6], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((self.fcn_dim[7], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))

        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, self.fcn_dim[6])), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, self.fcn_dim[7])), requires_grad=True))

        # hyper-network design
        hyper_dims += [self.k * self.k]
        input_dim = self.input_dim
        hyper_layers = []
        for i_dim in hyper_dims:
            hyper_layers.append(nn.Linear(input_dim, i_dim))
            hyper_layers.append(nn.BatchNorm1d(i_dim))
            hyper_layers.append(nn.ReLU())
            hyper_layers.append(nn.Dropout(p=0))
            input_dim = i_dim
        self.hyper_net = nn.Sequential(*hyper_layers)

        # adapter initiation
        self.b_list = nn.ParameterList() # bias
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((self.fcn_dim[6])), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((self.fcn_dim[7])), requires_grad=True))

        self.gamma1 = nn.Parameter(torch.ones(self.fcn_dim[6])) # domain norm parameters
        self.bias1 = nn.Parameter(torch.zeros(self.fcn_dim[6]))
        self.gamma2 = nn.Parameter(torch.ones(self.fcn_dim[7]))
        self.bias2 = nn.Parameter(torch.zeros(self.fcn_dim[7]))
        self.eps = 1e-5

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # hyper_network_out
            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # Representation matrix
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear

            domain_input = model_list[1](domain_input)  # bn

            domain_input = model_list[2](domain_input)  # relu    B * m



            domain_input = model_list[3](domain_input)  # linear

            domain_input = model_list[4](domain_input)  # bn

            domain_input = model_list[5](domain_input)  # relu




            domain_input = model_list[6](domain_input)  # linear

            domain_input = model_list[7](domain_input)  # bn

            domain_input = model_list[8](domain_input)  # relu





            domain_input = model_list[9](domain_input)  # linear

            domain_input = model_list[10](domain_input)  # bn

            domain_input = model_list[11](domain_input)  # relu




            domain_input = model_list[12](domain_input)  # linear

            domain_input = model_list[13](domain_input)  # bn

            domain_input = model_list[14](domain_input)  # relu




            domain_input = model_list[15](domain_input)  # linear

            domain_input = model_list[16](domain_input)  # bn

            domain_input = model_list[17](domain_input)  # relu

            # First Adapter-cell

            # Adapter layer-1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn',self.u[0] , hyper_out,self.v[0])
            b1 = self.b_list[0]
            tmp_out = torch.einsum('bf,bfj->bj',domain_input,w1)
            tmp_out += b1

            # Adapter layer-2: non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up - projection
            w2 = torch.einsum('mi,bij,jn->bmn',self.u[1] , hyper_out,self.v[1])
            b2 = self.b_list[1]
            tmp_out = torch.einsum('bf,bfj->bj',tmp_out,w2)
            tmp_out += b2

            # Adpater layer-4: Domain norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1

            # Adapter: short-cut
            domain_input = out+domain_input


            domain_input = model_list[18](domain_input)  # linear

            domain_input = model_list[19](domain_input)  # bn

            domain_input = model_list[20](domain_input)  # relu

            # Second Adapter-cell

            # Adapter layer-1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn', self.u[2], hyper_out, self.v[2])
            b1 = self.b_list[2]
            tmp_out = torch.einsum('bf,bfj->bj', domain_input, w1)
            tmp_out += b1

            # Adapter layer-2: non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up - projection
            w2 = torch.einsum('mi,bij,jn->bmn', self.u[3], hyper_out, self.v[3])
            b2 = self.b_list[3]
            tmp_out = torch.einsum('bf,bfj->bj', tmp_out, w2)
            tmp_out += b2

            # Adpater layer-4: Domain norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma2 * x_norm + self.bias2

            # Adapter: short-cut
            domain_input = out + domain_input


            domain_input = model_list[21](domain_input) # linear

            domain_input = self.sig(domain_input)       # relu

            out_l.append(domain_input)

        final = torch.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out_l[d], final)

        return final.squeeze(1)


class MLP_adap_2_layer_1_adp(nn.Module):
    # 2 layers MLP with 1 adapter cells
    def __init__(self, features, domain_num, fcn_dims, hyper_dims, k ):
        super().__init__()
        self.features = features
        # print("self.features",self.features)
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)
        #self.embedding = EmbeddingLayerHDRO(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()
        for d in range(domain_num):
            domain_specific = nn.ModuleList()
            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())
            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))

            self.layer_list.append(domain_specific)

        # instance matrix initiation
        self.k = k
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        # u,v initiation
        self.u.append(Parameter(torch.ones((self.fcn_dim[2], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))

        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, self.fcn_dim[2])), requires_grad=True))

        # hypernwt work
        hyper_dims += [self.k * self.k]
        input_dim = self.input_dim
        hyper_layers = []
        for i_dim in hyper_dims:
            hyper_layers.append(nn.Linear(input_dim, i_dim))
            hyper_layers.append(nn.BatchNorm1d(i_dim))
            hyper_layers.append(nn.ReLU())
            hyper_layers.append(nn.Dropout(p=0))
            input_dim = i_dim
        self.hyper_net = nn.Sequential(*hyper_layers)

        # Adapter parameters
        self.b_list = nn.ParameterList()
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((self.fcn_dim[2])), requires_grad=True))

        self.gamma1 = nn.Parameter(torch.ones(self.fcn_dim[2]))
        self.bias1 = nn.Parameter(torch.zeros(self.fcn_dim[2]))
        self.eps = 1e-5

    def get_emb_dict(self):

        return self.embedding

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        #print("self.domain_indicator", self.embedding.embed_dict['domain_indicator'].weight.data)
        # print(" self.domain_indicator0", self.embedding.embed_dict['domain_indicator'].weight.data)
        # exit()
        mask = []
        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            domain_input = emb
            # hyper-network output
            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # Representation matrix
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear

            domain_input = model_list[1](domain_input)  # bn

            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear

            domain_input = model_list[4](domain_input)  # bn

            domain_input = model_list[5](domain_input)  # relu
            # Adapter cell
            # Adapter layer-1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn', self.u[0], hyper_out, self.v[0])
            b1 = self.b_list[0]
            tmp_out = torch.einsum('bf,bfj->bj', domain_input, w1)
            tmp_out += b1
            # Adapter layer-2: Non-linear
            tmp_out = self.sig(tmp_out)
            # Adapter layer-3: Up projection
            w2 = torch.einsum('mi,bij,jn->bmn', self.u[1], hyper_out, self.v[1])
            b2 = self.b_list[1]
            tmp_out = torch.einsum('bf,bfj->bj', tmp_out, w2)
            tmp_out += b2
            # Adapter layer-4: Domain norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1
            # Adapter: Short-cut
            domain_input = out + domain_input

            domain_input = model_list[6](domain_input)
            domain_input = self.sig(domain_input)
            # exit()
            out_l.append(domain_input)
        #print("======================================================")
        #exit()
        final = torch.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out_l[d], final)

        return final.squeeze(1)



class Mlp_2_Layer(nn.Module):
    # 2-layres Mlp model
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()

        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out = []

        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_7_Layer(nn.Module):
    # 7-layers Mlp model
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()
        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], self.fcn_dim[3]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[3]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[3], self.fcn_dim[4]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[4]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[4], self.fcn_dim[5]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[5]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[5], self.fcn_dim[6]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[6]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[6], self.fcn_dim[7]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[7]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[7], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)  # linear
            domain_input = model_list[7](domain_input)  # bn
            domain_input = model_list[8](domain_input)  # relu

            domain_input = model_list[9](domain_input)  # linear
            domain_input = model_list[10](domain_input)  # bn
            domain_input = model_list[11](domain_input)  # relu

            domain_input = model_list[12](domain_input)  # linear
            domain_input = model_list[13](domain_input)  # bn
            domain_input = model_list[14](domain_input)  # relu

            domain_input = model_list[15](domain_input)  # linear
            domain_input = model_list[16](domain_input)  # bn
            domain_input = model_list[17](domain_input)  # relu

            domain_input = model_list[18](domain_input)  # linear
            domain_input = model_list[19](domain_input)  # bn
            domain_input = model_list[20](domain_input)  # relu

            domain_input = model_list[21](domain_input)
            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

class Mlp_2_Layer_SharedBottom(nn.Module):
    # 2-layres SharedBottom model
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)
        self.bottom = MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)
        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()

        for d in range(domain_num):

            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        fea = self.bottom(emb)
        mask = []

        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            domain_input = fea

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_MMoE(nn.Module):
    # 2-layres SharedBottom model
    """
       A pytorch implementation of MMoE Model.

       Reference:
           Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
       """
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)
        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")
        self.expert_num = 4
        self.expert = torch.nn.ModuleList(
            [MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2) for i in range(self.expert_num)])
        self.gate = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(self.input_dim,self.expert_num), torch.nn.Softmax(dim=1)) for i in range(self.domain_num)])

        # self.bottom = MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)

        self.layer_list = nn.ModuleList()

        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.domain_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.domain_num)]
        # fea = self.bottom(emb)
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = task_fea[d]

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_PLE(nn.Module):
    # 2-layres SharedBottom model
    """
       A pytorch implementation of PLE Model.

       Reference:
           Tang, Hongyan, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. RecSys 2020.
       """
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.embedding = EmbeddingLayer(features)
        self.domain_num = domain_num

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        # MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)
        self.bottom_mlp_dims= [256, 113]
        self.shared_expert_num = 2
        self.specific_expert_num = 2
        self.layers_num = len(self.bottom_mlp_dims)

        self.task_experts = [[0] * self.domain_num for _ in range(self.layers_num)]
        self.task_gates = [[0] * self.domain_num for _ in range(self.layers_num)]
        self.share_experts = [0] * self.layers_num
        self.share_gates = [0] * self.layers_num
        for i in range(self.layers_num):
            input_dim = self.input_dim if 0 == i else self.bottom_mlp_dims[i]
            self.share_experts[i] = torch.nn.ModuleList(
                [MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2) for k in
                 range(self.shared_expert_num)])
            self.share_gates[i] = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.shared_expert_num + domain_num * self.specific_expert_num), torch.nn.Softmax(dim=1))

            for j in range(domain_num):
                self.task_experts[i][j] = torch.nn.ModuleList(
                    [MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2) for k in range(self.specific_expert_num)])
                self.task_gates[i][j] = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, self.shared_expert_num + self.specific_expert_num), torch.nn.Softmax(dim=1))

            self.task_experts[i] = torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = torch.nn.ModuleList(self.task_gates[i])

        self.task_experts = torch.nn.ModuleList(self.task_experts)
        self.task_gates = torch.nn.ModuleList(self.task_gates)
        self.share_experts = torch.nn.ModuleList(self.share_experts)
        self.share_gates = torch.nn.ModuleList(self.share_gates)

        # self.bottom = MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)
        # == self.tower
        self.layer_list = nn.ModuleList()

        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        task_fea = [emb for i in range(self.domain_num + 1)] # task1 input ,task2 input,..taskn input, share_expert input

        for i in range(self.layers_num):
            share_output = [expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]]
            task_output_list = []
            for j in range(self.domain_num):
                task_output = [expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_ouput = torch.cat(task_output + share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)
            if i != self.layers_num - 1:  # 最后一层不需要计算share expert 的输出
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_ouput = torch.cat(task_output_list + share_output, dim=1)
                task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)

        # fea = self.bottom(emb)
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = task_fea[d]

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_AITM(nn.Module):
    """
       A pytorch implementation of Adaptive Information Transfer Multi-task Model.

       Reference:
           Xi, Dongbo, et al. Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising. KDD 2021.
       """
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.embedding = EmbeddingLayer(features)
        self.bottom_mlp_dims = [256, 113]
        self.domain_num = domain_num
        self.hidden_dim = self.bottom_mlp_dims[-1]

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.g = torch.nn.ModuleList(
            [torch.nn.Linear(self.bottom_mlp_dims[-1], self.bottom_mlp_dims[-1]) for i in range(domain_num - 1)])
        self.h1 = torch.nn.Linear(self.bottom_mlp_dims[-1], self.bottom_mlp_dims[-1])
        self.h2 = torch.nn.Linear(self.bottom_mlp_dims[-1], self.bottom_mlp_dims[-1])
        self.h3 = torch.nn.Linear(self.bottom_mlp_dims[-1], self.bottom_mlp_dims[-1])

        self.bottom = torch.nn.ModuleList(
            [MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2) for i in
             range(domain_num)])
        # MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)

        # self.bottom = MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)

        # == self.tower
        self.layer_list = nn.ModuleList()

        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        fea = [self.bottom[i](emb) for i in range(self.domain_num)]

        for i in range(1, self.domain_num):
            p = self.g[i - 1](fea[i - 1]).unsqueeze(1)
            q = fea[i].unsqueeze(1)
            x = torch.cat([p, q], dim=1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            fea[i] = torch.sum(
                torch.nn.functional.softmax(torch.sum(K * Q, 2, True) / np.sqrt(self.hidden_dim), dim=1) * V, 1)

        # fea = self.bottom(emb)
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = fea[d]

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_STAR(nn.Module):
    # 2-layres SharedBottom model
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)
        self.bottom = MLP(self.input_dim, output_layer=False,dims=[256, 113], dropout=0.2)
        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")
        self.layer_list = nn.ModuleList()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))

            self.layer_list.append(domain_specific)

        # star
        self.hidden_units=[113, 113,113]
        self.activation = nn.ReLU()
        # self.activation = nn.Softmax()
        self.shared_kernels = nn.ParameterList([nn.Parameter(torch.randn(self.hidden_units[i], self.hidden_units[i + 1]))
                                                for i in range(len(self.hidden_units) - 1)])

        self.shared_bias = nn.ParameterList([nn.Parameter(torch.randn(self.hidden_units[i+1]))
                                             for i in range(len(self.hidden_units) - 1)])

        self.domain_kernels = [[nn.Parameter(torch.randn(self.hidden_units[i], self.hidden_units[i + 1]))
                                for i in range(len(self.hidden_units) - 1)] for _ in range(domain_num)]

        self.domain_bias = [[nn.Parameter(torch.randn(self.hidden_units[i+1]))
                             for i in range(len(self.hidden_units) - 1)] for _ in range(domain_num)]

        self.activation_layers = nn.ModuleList(
            [self.activation for _ in range(len(self.hidden_units) - 1)])

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        fea = self.bottom(emb)
        domain_indicator = torch.nn.functional.one_hot(domain_id.squeeze(), 3)
        inputs = fea

        output_list=[inputs.clone() for _ in range(self.domain_num)]
        for i in range(len(self.hidden_units)-1):
            for j in range (self.domain_num):
                # print("self.shared_kernels[i]",self.shared_kernels[i].device)
                # print("self.domain_kernels[j][i]", self.domain_kernels[j][i].device) .to('cuda:2')
                # exit()
                output_list[j] = F.linear(output_list[j],
                                          self.shared_kernels[i] * self.domain_kernels[j][i].to(self.device),
                                          bias=self.shared_bias[i] + self.domain_bias[j][i].to(self.device))
                output_list[j] = self.activation_layers[i](output_list[j])

        output = torch.stack(output_list, dim=1) * domain_indicator.unsqueeze(-1)
        output = torch.sum(output, dim=1)

        fea=output
        # print("=================")
        mask = []

        out = []

        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            domain_input = fea

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            # print("domain_input",domain_input.size())
            # exit()
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)
        # print("out",out) # 3
        # exit()
        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)

        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_SharedBottom_kuairand(nn.Module):
    # 2-layres SharedBottom model
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)
        self.bottom = MLP(self.input_dim, output_layer=False, dims=[256, 49], dropout=0.2)
        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()

        for d in range(domain_num):

            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        fea = self.bottom(emb)
        mask = []

        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            domain_input = fea

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_MMoE_kuairand(nn.Module):
    # 2-layres SharedBottom model
    """
       A pytorch implementation of MMoE Model.

       Reference:
           Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
       """
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)
        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")
        self.expert_num = 4
        self.expert = torch.nn.ModuleList(
            [MLP(self.input_dim, output_layer=False, dims=[256, 49], dropout=0.2) for i in range(self.expert_num)])
        self.gate = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(self.input_dim,self.expert_num), torch.nn.Softmax(dim=1)) for i in range(self.domain_num)])

        # self.bottom = MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)

        self.layer_list = nn.ModuleList()

        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.domain_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.domain_num)]
        # fea = self.bottom(emb)
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = task_fea[d]

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_PLE_kuairand(nn.Module):
    # 2-layres SharedBottom model
    """
       A pytorch implementation of PLE Model.

       Reference:
           Tang, Hongyan, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. RecSys 2020.
       """
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.embedding = EmbeddingLayer(features)
        self.domain_num = domain_num

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        # MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)
        self.bottom_mlp_dims= [256, 49]
        self.shared_expert_num = 2
        self.specific_expert_num = 2
        self.layers_num = len(self.bottom_mlp_dims)

        self.task_experts = [[0] * self.domain_num for _ in range(self.layers_num)]
        self.task_gates = [[0] * self.domain_num for _ in range(self.layers_num)]
        self.share_experts = [0] * self.layers_num
        self.share_gates = [0] * self.layers_num
        for i in range(self.layers_num):
            input_dim = self.input_dim if 0 == i else self.bottom_mlp_dims[i]
            self.share_experts[i] = torch.nn.ModuleList(
                [MLP(self.input_dim, output_layer=False, dims=[256, 49], dropout=0.2) for k in
                 range(self.shared_expert_num)])
            self.share_gates[i] = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.shared_expert_num + domain_num * self.specific_expert_num), torch.nn.Softmax(dim=1))

            for j in range(domain_num):
                self.task_experts[i][j] = torch.nn.ModuleList(
                    [MLP(self.input_dim, output_layer=False, dims=[256, 49], dropout=0.2) for k in range(self.specific_expert_num)])
                self.task_gates[i][j] = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, self.shared_expert_num + self.specific_expert_num), torch.nn.Softmax(dim=1))

            self.task_experts[i] = torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = torch.nn.ModuleList(self.task_gates[i])

        self.task_experts = torch.nn.ModuleList(self.task_experts)
        self.task_gates = torch.nn.ModuleList(self.task_gates)
        self.share_experts = torch.nn.ModuleList(self.share_experts)
        self.share_gates = torch.nn.ModuleList(self.share_gates)

        # self.bottom = MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)
        # == self.tower
        self.layer_list = nn.ModuleList()

        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        task_fea = [emb for i in range(self.domain_num + 1)] # task1 input ,task2 input,..taskn input, share_expert input

        for i in range(self.layers_num):
            share_output = [expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]]
            task_output_list = []
            for j in range(self.domain_num):
                task_output = [expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_ouput = torch.cat(task_output + share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)
            if i != self.layers_num - 1:  # 最后一层不需要计算share expert 的输出
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_ouput = torch.cat(task_output_list + share_output, dim=1)
                task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)

        # fea = self.bottom(emb)
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = task_fea[d]

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_AITM_kuairand(nn.Module):
    """
       A pytorch implementation of Adaptive Information Transfer Multi-task Model.

       Reference:
           Xi, Dongbo, et al. Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising. KDD 2021.
       """
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.embedding = EmbeddingLayer(features)
        self.bottom_mlp_dims = [256, 49]
        self.domain_num = domain_num
        self.hidden_dim = self.bottom_mlp_dims[-1]

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.g = torch.nn.ModuleList(
            [torch.nn.Linear(self.bottom_mlp_dims[-1], self.bottom_mlp_dims[-1]) for i in range(domain_num - 1)])
        self.h1 = torch.nn.Linear(self.bottom_mlp_dims[-1], self.bottom_mlp_dims[-1])
        self.h2 = torch.nn.Linear(self.bottom_mlp_dims[-1], self.bottom_mlp_dims[-1])
        self.h3 = torch.nn.Linear(self.bottom_mlp_dims[-1], self.bottom_mlp_dims[-1])

        self.bottom = torch.nn.ModuleList(
            [MLP(self.input_dim, output_layer=False, dims=[256, 49], dropout=0.2) for i in
             range(domain_num)])
        # MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)

        # self.bottom = MLP(self.input_dim, output_layer=False, dims=[256, 113], dropout=0.2)

        # == self.tower
        self.layer_list = nn.ModuleList()

        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        fea = [self.bottom[i](emb) for i in range(self.domain_num)]

        for i in range(1, self.domain_num):
            p = self.g[i - 1](fea[i - 1]).unsqueeze(1)
            q = fea[i].unsqueeze(1)
            x = torch.cat([p, q], dim=1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            fea[i] = torch.sum(
                torch.nn.functional.softmax(torch.sum(K * Q, 2, True) / np.sqrt(self.hidden_dim), dim=1) * V, 1)

        # fea = self.bottom(emb)
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = fea[d]

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding

class Mlp_2_Layer_STAR_kuairand(nn.Module):
    # 2-layres STAR model
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)
        self.bottom = MLP(self.input_dim, output_layer=False,dims=[256, 49], dropout=0.2)
        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")
        self.layer_list = nn.ModuleList()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))

            self.layer_list.append(domain_specific)

        # star
        self.hidden_units=[49,49,49]
        self.activation = nn.ReLU()
        # self.activation = nn.Softmax()

        self.shared_kernels = nn.ParameterList([nn.Parameter(torch.randn(self.hidden_units[i], self.hidden_units[i + 1]))
                                                for i in range(len(self.hidden_units) - 1)])

        self.shared_bias = nn.ParameterList([nn.Parameter(torch.randn(self.hidden_units[i+1]))
                                             for i in range(len(self.hidden_units) - 1)])

        self.domain_kernels = [[nn.Parameter(torch.randn(self.hidden_units[i], self.hidden_units[i + 1]))
                                for i in range(len(self.hidden_units) - 1)] for _ in range(domain_num)]
        # print("11111", self.domain_kernels[0])
        # exit()
        self.domain_bias = [[nn.Parameter(torch.randn(self.hidden_units[i+1]))
                             for i in range(len(self.hidden_units) - 1)] for _ in range(domain_num)]

        self.activation_layers = nn.ModuleList(
            [self.activation for _ in range(len(self.hidden_units) - 1)])

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        fea = self.bottom(emb)
        domain_indicator = torch.nn.functional.one_hot(domain_id.squeeze(), self.domain_num)
        inputs = fea

        output_list=[inputs.clone() for _ in range(self.domain_num)]


        for i in range(len(self.hidden_units)-1):
            for j in range (self.domain_num):
                output_list[j] = F.linear(output_list[j],
                                          self.shared_kernels[i] * self.domain_kernels[j][i].to(self.device), # .to('cuda:2')
                                          bias=self.shared_bias[i] + self.domain_bias[j][i].to(self.device))
                output_list[j] = self.activation_layers[i](output_list[j])

        output = torch.stack(output_list, dim=1) * domain_indicator.unsqueeze(-1)
        output = torch.sum(output, dim=1)

        fea=output
        mask = []

        out = []

        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            domain_input = fea

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn

            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)

            domain_input = self.sig(domain_input)

            out.append(domain_input)
        # print("out",out) # 3
        # exit()
        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)

        return final.squeeze(1)

    def get_emb_dict(self):
        return self.embedding