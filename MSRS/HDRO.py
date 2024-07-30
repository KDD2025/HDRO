import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from MSRS.models.multi_domain import Mlp_7_Layer, Mlp_2_Layer, MLP_adap_2_layer_1_adp, DCN_MD, DCN_MD_adp, WideDeep_MD, WideDeep_MD_adp
from MSRS.models.multi_domain.adapter import Mlp_2_Layer_SharedBottom,Mlp_2_Layer_MMoE,Mlp_2_Layer_PLE,Mlp_2_Layer_AITM,Mlp_2_Layer_STAR
from MSRS.models.multi_domain.adapter import Mlp_2_Layer_SharedBottom_kuairand,Mlp_2_Layer_MMoE_kuairand,Mlp_2_Layer_PLE_kuairand,Mlp_2_Layer_AITM_kuairand,Mlp_2_Layer_STAR_kuairand
from MSRS.Utility.tools import enlargeSeq,seq2str
import numpy as np
from scipy.stats import norm, uniform
from scipy import integrate
import random

class Encoder_Decoder(nn.Module):
    def __init__(self,
                 k_class,
                 seq_num,
                 seq_dict,
                 seq_emb_dim
                 ):
        super(Encoder_Decoder, self).__init__()
        self.seq_num = seq_num
        self.seq_dict = seq_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.K = k_class
        self.d = 16
        self.d1 = 64
        self.d2 = 32
        self.d3 = 64
        self.seq_weight = nn.Embedding(self.seq_num,embedding_dim=self.K)
        self.M = nn.Parameter(torch.randn((self.d2, self.K)))
        self.V1 = nn.Linear(self.d, self.d1, bias=True)
        self.V2 = nn.Linear(self.d1, self.d2, bias=True)
        self.V3 = nn.Linear(self.d2, self.d3, bias=True)
        self.V4 = nn.Linear(self.d3, self.d, bias=True)
        self.mse = nn.MSELoss()
        self.apply(self._init_weights)
        # self.print_cluster_information()

    def forward(self, seq_cpu,seq_emb): # x_dict,embs4cluster

        seqEmb = torch.sum(seq_emb,dim=-2) # [batch_size,embedding_size] # [400,16]

        encoder_embedding = self.cluster_encoder(seqEmb) # [batch_size,d2] [400,32]
        decoder_embedding = self.cluster_decoder(encoder_embedding) # [batch_size,embedding_size] [400,16]

        batch_seq_prob = self.get_probability(seq_cpu) # [batch_size,k]
        center_embedding = torch.matmul(batch_seq_prob,self.M.transpose(0, 1)) # [batch_size,d2]
        cluster_loss = self.mse(encoder_embedding,center_embedding)

        reconstruction_loss =self.mse(seqEmb, decoder_embedding)
        loss = cluster_loss+reconstruction_loss
        return loss

    def get_probability(self,seq_cpu):
        seq_index_list = self.get_seq_index(seq_cpu)
        seq_index_list = torch.tensor(seq_index_list).unsqueeze(-1).cuda()
        batch_seq_weight = self.seq_weight(seq_index_list).squeeze(1)
        # else:
        #     seq_index_list = torch.tensor(seq_index_list).unsqueeze(-1).cuda()
        #     batch_seq_weight = self.seq_weight(seq_index_list).squeeze(1).cuda()
        return torch.softmax(batch_seq_weight, dim=-1)

    def x_to_seq(self,x_dict):
        user_id = x_dict['user_id']
        item_id = x_dict['movie_id']
        # user_list = user_id.T.numpy().tolist()
        # item_list = item_id.T.numpy().tolist()
        # user_list = user_id.T.cpu().numpy().tolist()
        # item_list = item_id.T.cpu().numpy().tolist()
        user_list = user_id.tolist()
        item_list = item_id.tolist()

        x_seq=[]
        for i in range(len(user_id)):
            user_item = []
            user = user_list[i]
            item = item_list[i]
            user_item.append(user)
            user_item.append(item)
            x_seq.append(user_item)

        return x_seq
        # return torch.tensor(x_seq)

    def get_seq_index(self,seq):
        seq =self.x_to_seq(seq)
        seq_list = seq
        seq_index_list =[]
        for sequence in seq_list:
            str_seq = seq2str(sequence)
            seq_index = self.seq_dict[str_seq]
            seq_index_list.append(seq_index)
        return seq_index_list

    def cluster_encoder(self, seq_emb):
        v_v1_embedding = self.V1(seq_emb)  # [batch_size,d1]
        v_sigmoid = torch.sigmoid(v_v1_embedding)
        v_v2_embedding = self.V2(v_sigmoid)  # [batch_size,d2]
        encoder_embedding = torch.softmax(v_v2_embedding / 1.0, dim=-1)
        return encoder_embedding

    def cluster_decoder(self,encoder_embedding):
        # encoder_embedding [batch_size,d2]
        v_v3_embedding = self.V3(encoder_embedding)  # [batch_size,d3]
        v_sigmoid = torch.sigmoid(v_v3_embedding)
        cluster_decoder_embedding = self.V4(v_sigmoid)  # [batch_size,d]
        return cluster_decoder_embedding

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def saveDict(self):
        dir = os.getcwd()
        fileName ='seq_prob_dict.txt'
        savePath = dir+"/"+fileName
        seq_class_prob= torch.softmax(self.seq_weight, dim=-1)
        _, indices = torch.topk(seq_class_prob, 1, -1)
        class_dict= {}
        for i in range(len(indices)):
            class_dict.update({str(i):str(indices[i][0].item())})
        new_seq_dict = {}
        for key in self.seq_dict.keys():
            seq_index = self.seq_dict[key]
            seq_class = class_dict[str(seq_index)]
            if key not in new_seq_dict.keys():
                new_seq_dict.update({key:seq_class})
        return new_seq_dict

    def outSeqDict(self,fileName):
        savePath = os.getcwd()+"/"+fileName
        seq_class_prob= torch.softmax(self.seq_weight, dim=-1)
        _, indices = torch.topk(seq_class_prob, 1, -1)
        class_dict= {}
        for i in range(len(indices)):
            class_dict.update({str(i):str(indices[i][0].item())})
        with open(savePath,'w') as fout:
            for key in self.seq_dict.keys():
                seq_index = self.seq_dict[key]
                seq_class = class_dict[str(seq_index)]
                outline = key+"\tclass:"+seq_class+"\n"
                fout.write(outline)

    def saveSeqProb(self,infomessage,global_w,t_w):
        dir = os.getcwd()
        fileName = 'seq_prob_dict.txt'
        savePath = dir + "/" + fileName
        fout = open(savePath, 'a')
        global_w_soft = torch.nn.functional.softmax(global_w/t_w, dim=1)
        print(infomessage,file=fout)
        for key in self.seq_dict.keys():
            seq_index = torch.tensor(int(self.seq_dict[key]))
            seq_index = seq_index.to(self.device).unsqueeze(-1)
            seq_prob = torch.softmax(self.seq_weight(seq_index).squeeze(1), dim = -1)
            print("seq: ",key,file=fout)
            print("prob: ",seq_prob,file=fout)
            print("weight: ",torch.sum(seq_prob*global_w_soft),file=fout)
            print("\n",file=fout)
        print("global_w:",global_w,file=fout)
        print("global_w_soft:",global_w_soft,file=fout)
        fout.close()

class HDRO_learning_w(nn.Module):
    def __init__(self,baseModel,
                 num_scenarios,
                 num_cluster_class,
                 n_items,
                 dense_feas,
                 sparse_feas,
                 domain_num,
                 seq_num,
                 seq_dict,
                 alpha,
                 beta,
                 lambda_w,
                 pre_dis
                 ):
        super(HDRO_learning_w, self).__init__()
        self.num_scenarios=num_scenarios
        self.seq_class = num_cluster_class
        self.n_items = n_items
        self.features = dense_feas+sparse_feas
        self.seq_num = seq_num
        self.seq_dict = seq_dict
        self.alpha =alpha
        self.beta = beta
        self.lambda_w=lambda_w
        self.pre_dis =pre_dis
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.V = (torch.nn.Parameter(data=torch.rand(1, num_cluster_class), requires_grad=True))
        #self.V_pre = [random.uniform(0, 1) for _ in range(num_cluster_class)]
        self.V_pre = np.full(num_cluster_class, 1 / num_cluster_class).tolist()
        # self.V = (torch.nn.Parameter(torch.full((num_seq,), (num_seq - 1) / 2.0 + 1), requires_grad=True))
        self.V_pre_2 = self.observe_distribution_v(self.seq_class)
        # self.V_pre_2=[0.0161, 0.0182, 0.0158, 0.02173, 0.01422, 0.01609, 0.01503, 0.01802, 0.01947, 0.01402, 0.01375, 0.01669, 0.01435, 0.01492, 0.01911, 0.01807, 0.01649, 0.02316, 0.01252, 0.01758, 0.02133, 0.01551, 0.02571, 0.01476, 0.02112, 0.01553, 0.01258, 0.02038, 0.01492, 0.018, 0.0128, 0.01543, 0.01923, 0.01665, 0.02049, 0.01522, 0.01585, 0.01415, 0.01416, 0.01457, 0.01484, 0.02087, 0.01454, 0.01332, 0.01606, 0.01784, 0.01596, 0.02153, 0.01488, 0.01634, 0.01817, 0.01299, 0.01438, 0.01697, 0.0141, 0.01264, 0.0184, 0.01842, 0.0165, 0.01751]
        self.t_v = 1
        self.W = (torch.nn.Parameter(data=torch.rand(1, num_scenarios), requires_grad=True)) # 3
        self.W.data /= self.W.data.sum()
        self.remove_W = 1
        self.remove_V = 1
        self.W_pre = np.full(num_scenarios, 1/num_scenarios).tolist()
        self.W_pre_2 = [0.21070,0.59443,0.19486]
        # target_distribution = np.array([210747 / (210747 + 594559 + 194903), 594559 / (210747 + 594559 + 194903),
        #                                 194903 / (210747 + 594559 + 194903)])
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.criterion = torch.nn.BCELoss()  # default loss cross_entropy
        self.criterion_v = torch.nn.BCELoss(reduction='none')
        self.checkpoint = None
        if baseModel == 'mlp':
            self.baseModel = Mlp_2_Layer(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == 'mlp_adp':
            self.baseModel = MLP_adap_2_layer_1_adp(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128],
                                           hyper_dims=[64], k=35)
        elif baseModel == "dcn_md":
            self.baseModel = DCN_MD(features=dense_feas + sparse_feas,num_domains=domain_num ,n_cross_layers=2, mlp_params={"dims": [256, 128]})
        elif baseModel == "dcn_md_adp":
            self.baseModel = DCN_MD_adp(features=dense_feas + sparse_feas,num_domains=domain_num, n_cross_layers=2, k = 30, mlp_params={"dims": [256, 128]}, hyper_dims=[128])
        elif baseModel == "wd_md":
            self.baseModel = WideDeep_MD(wide_features=dense_feas,num_domains= domain_num, deep_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
        elif baseModel == "wd_md_adp":
            self.baseModel = WideDeep_MD_adp(wide_features=dense_feas,num_domains= domain_num, deep_features=sparse_feas,  k= 45,mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}, hyper_dims=[128])
        elif baseModel == "SharedBottom":
            self.baseModel = Mlp_2_Layer_SharedBottom(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "MMoE":
            self.baseModel = Mlp_2_Layer_MMoE(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "PLE":
            self.baseModel = Mlp_2_Layer_PLE(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "AITM":
            self.baseModel = Mlp_2_Layer_AITM(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "STAR":
            self.baseModel = Mlp_2_Layer_STAR(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])

        else:
            pass
        self.emb = self.baseModel.get_emb_dict()

        self.cluster = Encoder_Decoder(
            k_class = self.seq_class,
             seq_num = self.seq_num,
             seq_dict = self.seq_dict,
            seq_emb_dim = 1100
            )
    def observe_distribution_v(self, cluster_class):
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        data_path = '/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/ml-1m'
        df_train = pd.read_csv(data_path + '/train_dataset.csv')
        # df_val = pd.read_csv(data_path + '/val_dataset.csv')
        # df_test = pd.read_csv(data_path + '/test_dataset_0.8_0.8.csv')
        del df_train["label"]
        # del df_val["label"]
        # del df_test["label"]
        # df_train = df_train[:100]
        # df_val = df_val[:100]
        # df_test = df_test[:100]
        data = df_train

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        user_item_numeric = data[numeric_columns]

        # 使用 PCA 进行降维
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(user_item_numeric)

        # 将用户和物品嵌入连接，形成样本表示
        samples_representation = np.hstack((embedding[:, 0].reshape(-1, 1), embedding[:, 1].reshape(-1, 1)))

        data = samples_representation
        # 使用 K-Means 聚类将数据分成3类
        n_clusters = cluster_class
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)

        # 统计每个类别的样本个数
        counts_by_cluster = np.bincount(labels)

        # 计算正则化后的类别分布
        normalized_distribution = np.array(np.round(normalize(counts_by_cluster.reshape(1, -1), norm='l1'),5)).tolist()[0]

        return normalized_distribution
    def kl_divergence(self, p, q, epsilon=1e-8):
        p = p.cpu()
        p = torch.clamp(p, epsilon, 1 - epsilon)
        q = torch.clamp(q, epsilon, 1 - epsilon)
        return torch.sum(p * torch.log(p / q))

    def w_weight_loss_movielens(self, x_dict, unweighted_loss, inter_scores_scenario_1, inter_scores_scenario_2,
                             inter_scores_scenario_3):
        len_1 = len(inter_scores_scenario_1)
        len_2 = len(inter_scores_scenario_2)
        len_3 = len(inter_scores_scenario_3)
        if len_1 == 0:
            # new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            # new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

            #  weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_1 = weighted_loss_v_3.detach().clone().fill_(0.0000001).requires_grad_()
        elif len_2 == 0:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            # new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            # new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            # weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_2 = weighted_loss_v_3.detach().clone().fill_(0.0000001).requires_grad_()
        elif len_3 == 0:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            # new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            # new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            # weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_3 = weighted_loss_v_2.detach().clone().fill_(0.0000001).requires_grad_()
        else:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)

        return weighted_loss_v_1, weighted_loss_v_2, weighted_loss_v_3
    def forward(self, x_dict, y):
        self.emb = self.baseModel.get_emb_dict()
        embs4cluster = self.emb(x_dict,self.features)

        cluster_loss = self.cluster(x_dict,embs4cluster)

        scores = self.baseModel(x_dict)

        #  add scenarios weight V W in training process
        unweighted_loss = self.criterion_v(scores.float(), y.float())

        scenarios_ids = [0, 1, 2]
        inter_scores_scenario_1 = []
        inter_scores_scenario_2 = []
        inter_scores_scenario_3 = []
        x_dict_list = x_dict["domain_indicator"].tolist()
        len_indicator = len(x_dict_list)

        for i in range(len_indicator):
            if x_dict["domain_indicator"].tolist()[i] == scenarios_ids[0]:
                inter_scores_scenario_1.append(i)
            elif x_dict["domain_indicator"].tolist()[i] == scenarios_ids[1]:
                inter_scores_scenario_2.append(i)
            elif x_dict["domain_indicator"].tolist()[i] == scenarios_ids[2]:
                inter_scores_scenario_3.append(i)

        if len(inter_scores_scenario_1) ==0:
            inter_scores_scenario_1= [0, 2]
        else:
            inter_scores_scenario_1= inter_scores_scenario_1

        if len(inter_scores_scenario_2) ==0:
            inter_scores_scenario_2= [1, 2]
        else:
            inter_scores_scenario_2= inter_scores_scenario_2

        if len(inter_scores_scenario_3) ==0:
            inter_scores_scenario_3= [1]
        else:
            inter_scores_scenario_3= inter_scores_scenario_3
        # weighted_loss_v_1, weighted_loss_v_2, weighted_loss_v_3 = self.w_weight_loss_movielens(x_dict, unweighted_loss,
        #                                                                                     inter_scores_scenario_1,
        #                                                                                     inter_scores_scenario_2,
        #                                                                                     inter_scores_scenario_3)
        new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
        new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
        new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
        # v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
        v_softmax = self.V
        seq_class_softmax = self.cluster.get_probability(x_dict)
        seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
        if self.alpha != 0:
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]
            # print("0000")
            # exit()
        else:
            new_seq_weight_scenario_1 = new_seq_weight_scenario_2=new_seq_weight_scenario_3=1
            # print("11111")
            # exit()
        weighted_loss_v_1 = torch.sum(new_seq_weight_scenario_1 * new_tensor_scenario_1)
        weighted_loss_v_2 = torch.sum(new_seq_weight_scenario_2 * new_tensor_scenario_2)
        weighted_loss_v_3 = torch.sum(new_seq_weight_scenario_3 * new_tensor_scenario_3)

        #weighted_loss_v_loss = torch.mean(weighted_loss_v_1+weighted_loss_v_2+weighted_loss_v_3)
        weighted_loss_w_distribution  = torch.stack([weighted_loss_v_1, weighted_loss_v_2, weighted_loss_v_3])
        if self.lambda_w != 0:
            weighted_loss_w_loss = torch.sum(self.W * weighted_loss_w_distribution)
        else:
            weighted_loss_w_loss = torch.sum(self.remove_W * weighted_loss_w_distribution)
        #weighted_loss = weighted_loss_v_loss
        #weighted_loss += weighted_loss_w_loss
        weighted_loss = weighted_loss_w_loss
        # unwighted_loss = self.criterion(scores.float(), y.float())
        # v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
        # seq_class_softmax = self.cluster.get_probability(x_dict)
        # seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
        # weighted_loss = torch.mean(seq_weight*unwighted_loss)

        # kl V loss
        if self.pre_dis == 1:
            with torch.no_grad():
                self.V_pre = torch.as_tensor(self.V_pre).clone()
            kl_V_loss = self.kl_divergence(self.V, self.V_pre)
            weighted_loss = weighted_loss + self.alpha * kl_V_loss

            # kl w loss
            with torch.no_grad():
                self.W_pre = torch.as_tensor(self.W_pre).clone()
            kl_W_loss = self.kl_divergence(self.W, self.W_pre)
            weighted_loss = weighted_loss + self.lambda_w * kl_W_loss
        else:
            with torch.no_grad():
                self.V_pre_2 = torch.as_tensor(self.V_pre_2).clone()
            kl_V_loss = self.kl_divergence(self.V, self.V_pre_2)
            weighted_loss = weighted_loss + self.alpha * kl_V_loss

            # kl w loss
            with torch.no_grad():
                self.W_pre_2 = torch.as_tensor(self.W_pre_2).clone()
            kl_W_loss = self.kl_divergence(self.W, self.W_pre_2)
            weighted_loss = weighted_loss + self.lambda_w * kl_W_loss

        return weighted_loss, self.beta * cluster_loss
class Encoder_Decoder_douban(nn.Module):
    def __init__(self,
                 k_class,
                 seq_num,
                 seq_dict,
                 seq_emb_dim
                 ):
        super(Encoder_Decoder_douban, self).__init__()
        self.seq_num = seq_num
        self.seq_dict = seq_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.K = k_class
        self.d =  16
        self.d1 = 64
        self.d2 = 32
        self.d3 = 64
        self.seq_weight = nn.Embedding(self.seq_num, embedding_dim=self.K)
        self.M = nn.Parameter(torch.randn((self.d2, self.K)))
        self.V1 = nn.Linear(self.d, self.d1, bias=True)
        self.V2 = nn.Linear(self.d1, self.d2, bias=True)
        self.V3 = nn.Linear(self.d2, self.d3, bias=True)
        self.V4 = nn.Linear(self.d3, self.d, bias=True)
        self.mse = nn.MSELoss()
        self.apply(self._init_weights)
        # self.print_cluster_information()

    def forward(self, seq_cpu, seq_emb):  # x_dict,embs4cluster

        seqEmb = torch.sum(seq_emb, dim=-2)  # [batch_size,embedding_size] # [400,16]
        encoder_embedding = self.cluster_encoder(seqEmb)  # [batch_size,d2] [400,32]
        decoder_embedding = self.cluster_decoder(encoder_embedding)  # [batch_size,embedding_size] [400,16]

        batch_seq_prob = self.get_probability(seq_cpu)  # [batch_size,k]
        center_embedding = torch.matmul(batch_seq_prob, self.M.transpose(0, 1))  # [batch_size,d2]
        # exit()
        cluster_loss = self.mse(encoder_embedding, center_embedding)
        reconstruction_loss = self.mse(seqEmb, decoder_embedding)
        # print("reconstruction_loss", reconstruction_loss)
        loss = cluster_loss + reconstruction_loss
        #exit()
        return loss

    def get_probability(self, seq_cpu):
        seq_index_list = self.get_seq_index(seq_cpu)
        seq_index_list = torch.tensor(seq_index_list).unsqueeze(-1).cuda()
        batch_seq_weight = self.seq_weight(seq_index_list).squeeze(1)
        # else:
        #     seq_index_list = torch.tensor(seq_index_list).unsqueeze(-1).cuda()
        #     batch_seq_weight = self.seq_weight(seq_index_list).squeeze(1).cuda()
        return torch.softmax(batch_seq_weight, dim=-1)

    def x_to_seq(self, x_dict):
        user_id = x_dict['user_id']
        item_id = x_dict['item_id']
        # user_list = user_id.T.numpy().tolist()
        # item_list = item_id.T.numpy().tolist()
        # user_list = user_id.T.cpu().numpy().tolist()
        # item_list = item_id.T.cpu().numpy().tolist()
        user_list = user_id.tolist()
        item_list = item_id.tolist()

        x_seq = []
        for i in range(len(user_id)):
            user_item = []
            user = user_list[i]
            item = item_list[i]
            user_item.append(user)
            user_item.append(item)
            x_seq.append(user_item)

        return x_seq
        # return torch.tensor(x_seq)

    def get_seq_index(self, seq):
        seq = self.x_to_seq(seq)
        seq_list = seq
        seq_index_list = []
        for sequence in seq_list:
            str_seq = seq2str(sequence)
            seq_index = self.seq_dict[str_seq]
            seq_index_list.append(seq_index)
        return seq_index_list

    def cluster_encoder(self, seq_emb):
        v_v1_embedding = self.V1(seq_emb)  # [batch_size,d1]
        v_sigmoid = torch.sigmoid(v_v1_embedding)
        v_v2_embedding = self.V2(v_sigmoid)  # [batch_size,d2]
        encoder_embedding = torch.softmax(v_v2_embedding / 1.0, dim=-1)
        return encoder_embedding

    def cluster_decoder(self, encoder_embedding):
        # encoder_embedding [batch_size,d2]
        v_v3_embedding = self.V3(encoder_embedding)  # [batch_size,d3]
        v_sigmoid = torch.sigmoid(v_v3_embedding)
        cluster_decoder_embedding = self.V4(v_sigmoid)  # [batch_size,d]
        return cluster_decoder_embedding

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def saveDict(self):
        dir = os.getcwd()
        fileName = 'seq_prob_dict.txt'
        savePath = dir + "/" + fileName
        seq_class_prob = torch.softmax(self.seq_weight, dim=-1)
        _, indices = torch.topk(seq_class_prob, 1, -1)
        class_dict = {}
        for i in range(len(indices)):
            class_dict.update({str(i): str(indices[i][0].item())})
        new_seq_dict = {}
        for key in self.seq_dict.keys():
            seq_index = self.seq_dict[key]
            seq_class = class_dict[str(seq_index)]
            if key not in new_seq_dict.keys():
                new_seq_dict.update({key: seq_class})
        return new_seq_dict

    def outSeqDict(self, fileName):
        savePath = os.getcwd() + "/" + fileName
        seq_class_prob = torch.softmax(self.seq_weight, dim=-1)
        _, indices = torch.topk(seq_class_prob, 1, -1)
        class_dict = {}
        for i in range(len(indices)):
            class_dict.update({str(i): str(indices[i][0].item())})
        with open(savePath, 'w') as fout:
            for key in self.seq_dict.keys():
                seq_index = self.seq_dict[key]
                seq_class = class_dict[str(seq_index)]
                outline = key + "\tclass:" + seq_class + "\n"
                fout.write(outline)

    def saveSeqProb(self, infomessage, global_w, t_w):
        dir = os.getcwd()
        fileName = 'seq_prob_dict.txt'
        savePath = dir + "/" + fileName
        fout = open(savePath, 'a')
        global_w_soft = torch.nn.functional.softmax(global_w / t_w, dim=1)
        print(infomessage, file=fout)
        for key in self.seq_dict.keys():
            seq_index = torch.tensor(int(self.seq_dict[key]))
            seq_index = seq_index.to(self.device).unsqueeze(-1)
            seq_prob = torch.softmax(self.seq_weight(seq_index).squeeze(1), dim=-1)
            print("seq: ", key, file=fout)
            print("prob: ", seq_prob, file=fout)
            print("weight: ", torch.sum(seq_prob * global_w_soft), file=fout)
            print("\n", file=fout)
        print("global_w:", global_w, file=fout)
        print("global_w_soft:", global_w_soft, file=fout)
        fout.close()

class HDRO_learning_w_douban(nn.Module):
    def __init__(self, baseModel,
                 num_scenarios,
                 num_cluster_class,
                 n_items,
                 dense_feas,
                 sparse_feas,
                 domain_num,
                 seq_num,
                 seq_dict,
                 alpha,
                 beta,
                 lambda_w,
                 pre_dis
                 ):
        super(HDRO_learning_w_douban, self).__init__()
        self.num_scenarios = num_scenarios
        self.seq_class = num_cluster_class
        self.n_items = n_items
        self.features = dense_feas + sparse_feas
        self.seq_num = seq_num
        self.seq_dict = seq_dict
        self.alpha = alpha
        self.beta = beta
        self.lambda_w = lambda_w
        self.pre_dis = pre_dis
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.V = (torch.nn.Parameter(data=torch.rand(1, num_cluster_class), requires_grad=True))
        self.V.data /= self.V.data.sum()
        # self.V_pre = [random.uniform(0, 1) for _ in range(num_cluster_class)]
        self.V_pre = np.full(num_cluster_class, 1 / num_cluster_class).tolist()
        # self.V = (torch.nn.Parameter(torch.full((num_seq,), (num_seq - 1) / 2.0 + 1), requires_grad=True))
        #self.V_pre_2 = self.observe_distribution_v(self.seq_class)
        self.V_pre_2= [0.00798, 0.02218, 0.01922, 0.00531, 0.02561, 0.01806, 0.01759, 0.02316, 0.03223, 0.01431, 0.0062, 0.01786, 0.00697, 0.00563, 0.01644, 0.01525, 0.03127, 0.0167, 0.0182, 0.02098, 0.01571, 0.01887, 0.00559, 0.00688, 0.01999, 0.00707, 0.01891, 0.02765, 0.0263, 0.03183, 0.02569, 0.03207, 0.02026, 0.00515, 0.0169, 0.01721, 0.00833, 0.00801, 0.01801, 0.02104, 0.00559, 0.01406, 0.006, 0.02254, 0.01166, 0.02254, 0.01631, 0.00799, 0.00697, 0.02335, 0.02651, 0.01904, 0.03204, 0.01729, 0.00836, 0.01995, 0.00531, 0.0089, 0.01498, 0.01798]
        # [0.00798, 0.02218, 0.01922, 0.00531, 0.02561, 0.01806, 0.01759, 0.02316, 0.03223, 0.01431, 0.0062, 0.01786, 0.00697, 0.00563, 0.01644, 0.01525, 0.03127, 0.0167, 0.0182, 0.02098, 0.01571, 0.01887, 0.00559, 0.00688, 0.01999, 0.00707, 0.01891, 0.02765, 0.0263, 0.03183, 0.02569, 0.03207, 0.02026, 0.00515, 0.0169, 0.01721, 0.00833, 0.00801, 0.01801, 0.02104, 0.00559, 0.01406, 0.006, 0.02254, 0.01166, 0.02254, 0.01631, 0.00799, 0.00697, 0.02335, 0.02651, 0.01904, 0.03204, 0.01729, 0.00836, 0.01995, 0.00531, 0.0089, 0.01498, 0.01798]
        # exit()
        self.t_v = 1
        self.W = (torch.nn.Parameter(data=torch.rand(1, num_scenarios), requires_grad=True))
        self.W.data /= self.W.data.sum()
        # self.W_pre = [random.uniform(0, 1) for _ in range(num_scenarios)]
        self.W_pre = np.full(num_scenarios, 1/num_scenarios).tolist()
        self.W_pre_2 = [0.05366, 0.07392, 0.87242]
        # target_distribution = np.array(
        #     [69709 / (69709 + 96041 + 1133420), 96041 / (69709 + 96041 + 1133420), 1133420 / (69709 + 96041 + 1133420)])
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.criterion = torch.nn.BCELoss()  # default loss cross_entropy
        self.criterion_v = torch.nn.BCELoss(reduction='none')
        self.checkpoint = None
        if baseModel == 'mlp':
            self.baseModel = Mlp_2_Layer(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == 'mlp_adp':
            self.baseModel = MLP_adap_2_layer_1_adp(dense_feas + sparse_feas, domain_num=domain_num,
                                                    fcn_dims=[256, 128],
                                                    hyper_dims=[64], k=35)
        elif baseModel == "dcn_md":
            self.baseModel = DCN_MD(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=2,
                                    mlp_params={"dims": [256, 128]})
        elif baseModel == "dcn_md_adp":
            self.baseModel = DCN_MD_adp(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=2,
                                        k=30, mlp_params={"dims": [256, 128]}, hyper_dims=[128])
        elif baseModel == "wd_md":
            self.baseModel = WideDeep_MD(wide_features=dense_feas, num_domains=domain_num, deep_features=sparse_feas,
                                         mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
        elif baseModel == "wd_md_adp":
            self.baseModel = WideDeep_MD_adp(wide_features=dense_feas, num_domains=domain_num,
                                             deep_features=sparse_feas, k=45,
                                             mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
                                             hyper_dims=[128])
        elif baseModel == "SharedBottom":
            self.baseModel = Mlp_2_Layer_SharedBottom_kuairand(dense_feas + sparse_feas, domain_num=domain_num,
                                                      fcn_dims=[256, 128])
        elif baseModel == "MMoE":
            self.baseModel = Mlp_2_Layer_MMoE_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "PLE":
            self.baseModel = Mlp_2_Layer_PLE_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "AITM":
            self.baseModel = Mlp_2_Layer_AITM_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "STAR":
            self.baseModel = Mlp_2_Layer_STAR_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])

        else:
            pass
        self.emb = self.baseModel.get_emb_dict()

        self.cluster = Encoder_Decoder_douban(
            k_class=self.seq_class,
            seq_num=self.seq_num,
            seq_dict=self.seq_dict,
            seq_emb_dim=1100
        )
    def observe_distribution_v(self, cluster_class):
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        data_path = '/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/douban'
        df_train = pd.read_csv(data_path + '/train_dataset.csv')
        # df_val = pd.read_csv(data_path + '/val_dataset.csv')
        # df_test = pd.read_csv(data_path + '/test_dataset_0.8_0.8.csv')
        del df_train["label"]
        # del df_val["label"]
        # del df_test["label"]
        # df_train = df_train[:100]
        # df_val = df_val[:100]
        # df_test = df_test[:100]
        data = df_train

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        user_item_numeric = data[numeric_columns]

        # 使用 PCA 进行降维
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(user_item_numeric)

        # 将用户和物品嵌入连接，形成样本表示
        samples_representation = np.hstack((embedding[:, 0].reshape(-1, 1), embedding[:, 1].reshape(-1, 1)))

        data = samples_representation
        # 使用 K-Means 聚类将数据分成3类
        n_clusters = cluster_class
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)

        # 统计每个类别的样本个数
        counts_by_cluster = np.bincount(labels)

        # 计算正则化后的类别分布
        normalized_distribution = np.array(np.round(normalize(counts_by_cluster.reshape(1, -1), norm='l1'),5)).tolist()[0]

        return normalized_distribution

    def kl_divergence(self,p, q, epsilon=1e-8):
        p = p.cpu()
        p = torch.clamp(p, epsilon, 1 - epsilon)
        q = torch.clamp(q, epsilon, 1 - epsilon)
        return torch.sum(p * torch.log(p / q))

    def w_weight_loss_douban(self,x_dict,unweighted_loss,inter_scores_scenario_1,inter_scores_scenario_2,inter_scores_scenario_3):
        len_1= len(inter_scores_scenario_1)
        len_2 = len(inter_scores_scenario_2)
        len_3 = len(inter_scores_scenario_3)
        if len_1 == 0:
            #new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            #new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

            #  weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_1 = weighted_loss_v_3.detach().clone().fill_(0.0000001).requires_grad_()
        elif len_2 ==0:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            #new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            #new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            #weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_2 = weighted_loss_v_3.detach().clone().fill_(0.0000001).requires_grad_()
        elif len_3 == 0:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            #new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            #new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            # weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_3 = weighted_loss_v_2.detach().clone().fill_(0.0000001).requires_grad_()
        else:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)

        return weighted_loss_v_1,weighted_loss_v_2,weighted_loss_v_3

    def forward(self, x_dict, y):
        self.emb = self.baseModel.get_emb_dict()
        embs4cluster = self.emb(x_dict, self.features)
        cluster_loss = self.cluster(x_dict, embs4cluster)
        scores = self.baseModel(x_dict)

        #  add scenarios weight V W in training process
        unweighted_loss = self.criterion_v(scores, y.float())
        # print("domain_indicator", x_dict["domain_indicator"])
        # scenarios_ids_num = list(set(x_dict["domain_indicator"].tolist()))
        scenarios_ids = [0,1,2]
        inter_scores_scenario_1 = []
        inter_scores_scenario_2 = []
        inter_scores_scenario_3 = []
        x_dict_list=x_dict["domain_indicator"].tolist()
        len_indicator = len(x_dict_list)
        for i in range(len_indicator):
            if x_dict_list[i] == scenarios_ids[0]:
                inter_scores_scenario_1.append(i)
            elif x_dict_list[i] == scenarios_ids[1]:
                inter_scores_scenario_2.append(i)
            elif x_dict_list[i] == scenarios_ids[2]:
                inter_scores_scenario_3.append(i)

        if len(inter_scores_scenario_1) ==0:
            inter_scores_scenario_1= [0, 2]
        else:
            inter_scores_scenario_1= inter_scores_scenario_1

        if len(inter_scores_scenario_2) ==0:
            inter_scores_scenario_2= [1, 2]
        else:
            inter_scores_scenario_2= inter_scores_scenario_2

        if len(inter_scores_scenario_3) ==0:
            inter_scores_scenario_3= [1]
        else:
            inter_scores_scenario_3= inter_scores_scenario_3

        new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
        new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
        new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]

        # print("self.V", self.V)
        # v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
        v_softmax = self.V
        # print("v_softmax",v_softmax)
        seq_class_softmax = self.cluster.get_probability(x_dict)
        seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
        new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
        new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
        new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]

        weighted_loss_v_1 = torch.sum(new_seq_weight_scenario_1 * new_tensor_scenario_1)
        weighted_loss_v_2 = torch.sum(new_seq_weight_scenario_2 * new_tensor_scenario_2)
        weighted_loss_v_3 = torch.sum(new_seq_weight_scenario_3 * new_tensor_scenario_3)
        # weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
        # weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
        # weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
        # print("77777")
        # weighted_loss_v_1, weighted_loss_v_2, weighted_loss_v_3 = self.w_weight_loss_douban(x_dict, unweighted_loss,
        #                                                                              inter_scores_scenario_1,
        #                                                                              inter_scores_scenario_2,
        #                                                                              inter_scores_scenario_3)

        # weighted_loss_v_loss = torch.mean(weighted_loss_v_1 + weighted_loss_v_2 + weighted_loss_v_3)
        weighted_loss_w_distribution = torch.stack([weighted_loss_v_1, weighted_loss_v_2, weighted_loss_v_3])

        weighted_loss_w_loss = torch.sum(self.W * weighted_loss_w_distribution)
        weighted_loss = weighted_loss_w_loss
        #weighted_loss = weighted_loss_v_loss
        #weighted_loss += weighted_loss_w_loss
        # unwighted_loss = self.criterion(scores.float(), y.float())
        # v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
        # seq_class_softmax = self.cluster.get_probability(x_dict)
        # seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
        # weighted_loss = torch.mean(seq_weight*unwighted_loss)

        # kl V loss
        if self.pre_dis == 1:
            with torch.no_grad():
                self.V_pre = torch.as_tensor(self.V_pre).clone()
            kl_V_loss = self.kl_divergence(self.V, self.V_pre)
            weighted_loss = weighted_loss + self.alpha * kl_V_loss

            # kl w loss
            with torch.no_grad():
                self.W_pre = torch.as_tensor(self.W_pre).clone()
            kl_W_loss = self.kl_divergence(self.W, self.W_pre)
            weighted_loss = weighted_loss + self.lambda_w * kl_W_loss
        else:

            with torch.no_grad():
                self.V_pre_2 = torch.as_tensor(self.V_pre_2).clone()
            kl_V_loss = self.kl_divergence(self.V, self.V_pre_2)
            weighted_loss = weighted_loss + self.alpha * kl_V_loss

            # kl w loss
            with torch.no_grad():
                self.W_pre_2 = torch.as_tensor(self.W_pre_2).clone()
            kl_W_loss = self.kl_divergence(self.W, self.W_pre_2)
            weighted_loss = weighted_loss + self.lambda_w * kl_W_loss

        return weighted_loss, self.beta * cluster_loss

class Encoder_Decoder_kuairand(nn.Module):
    def __init__(self,
                 k_class,
                 seq_num,
                 seq_dict,
                 seq_emb_dim
                 ):
        super(Encoder_Decoder_kuairand, self).__init__()
        self.seq_num = seq_num
        self.seq_dict = seq_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.K = k_class
        self.d = 16
        self.d1 = 64
        self.d2 = 32
        self.d3 = 64
        self.seq_weight = nn.Embedding(self.seq_num,embedding_dim=self.K)
        self.M = nn.Parameter(torch.randn((self.d2, self.K)))
        self.V1 = nn.Linear(self.d, self.d1, bias=True)
        self.V2 = nn.Linear(self.d1, self.d2, bias=True)
        self.V3 = nn.Linear(self.d2, self.d3, bias=True)
        self.V4 = nn.Linear(self.d3, self.d, bias=True)
        self.mse = nn.MSELoss()
        self.apply(self._init_weights)

    def forward(self, seq_cpu,seq_emb): # x_dict,embs4cluster

        seqEmb = torch.sum(seq_emb,dim=-2) # [batch_size,embedding_size] # [400,16]

        encoder_embedding = self.cluster_encoder(seqEmb) # [batch_size,d2] [400,32]
        decoder_embedding = self.cluster_decoder(encoder_embedding) # [batch_size,embedding_size] [400,16]

        batch_seq_prob = self.get_probability(seq_cpu) # [batch_size,k]
        center_embedding = torch.matmul(batch_seq_prob,self.M.transpose(0, 1)) # [batch_size,d2]
        cluster_loss = self.mse(encoder_embedding,center_embedding)

        reconstruction_loss =self.mse(seqEmb, decoder_embedding)
        loss = cluster_loss+reconstruction_loss
        return loss

    def get_probability(self,seq_cpu):
        seq_index_list = self.get_seq_index(seq_cpu)
        seq_index_list = torch.tensor(seq_index_list).unsqueeze(-1).cuda()
        batch_seq_weight = self.seq_weight(seq_index_list).squeeze(1)
        return torch.softmax(batch_seq_weight, dim=-1)

    def x_to_seq(self,x_dict):
        user_id = x_dict['user_id']
        item_id = x_dict['item_id']
        user_list = user_id.tolist()
        item_list = item_id.tolist()
        x_seq=[]
        for i in range(len(user_id)):
            user_item = []
            user = user_list[i]
            item = item_list[i]
            user_item.append(user)
            user_item.append(item)
            x_seq.append(user_item)

        return x_seq
        # return torch.tensor(x_seq)

    def get_seq_index(self,seq):
        seq =self.x_to_seq(seq)
        seq_list = seq
        seq_index_list =[]
        for sequence in seq_list:
            str_seq = seq2str(sequence)
            seq_index = self.seq_dict[str_seq]
            seq_index_list.append(seq_index)
        return seq_index_list

    def cluster_encoder(self, seq_emb):
        v_v1_embedding = self.V1(seq_emb)  # [batch_size,d1]
        v_sigmoid = torch.sigmoid(v_v1_embedding)
        v_v2_embedding = self.V2(v_sigmoid)  # [batch_size,d2]
        encoder_embedding = torch.softmax(v_v2_embedding / 1.0, dim=-1)
        return encoder_embedding

    def cluster_decoder(self,encoder_embedding):
        # encoder_embedding [batch_size,d2]
        v_v3_embedding = self.V3(encoder_embedding)  # [batch_size,d3]
        v_sigmoid = torch.sigmoid(v_v3_embedding)
        cluster_decoder_embedding = self.V4(v_sigmoid)  # [batch_size,d]
        return cluster_decoder_embedding

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def saveDict(self):
        dir = os.getcwd()
        fileName ='seq_prob_dict.txt'
        savePath = dir+"/"+fileName
        seq_class_prob= torch.softmax(self.seq_weight, dim=-1)
        _, indices = torch.topk(seq_class_prob, 1, -1)
        class_dict= {}
        for i in range(len(indices)):
            class_dict.update({str(i):str(indices[i][0].item())})
        new_seq_dict = {}
        for key in self.seq_dict.keys():
            seq_index = self.seq_dict[key]
            seq_class = class_dict[str(seq_index)]
            if key not in new_seq_dict.keys():
                new_seq_dict.update({key:seq_class})
        return new_seq_dict

    def outSeqDict(self,fileName):
        savePath = os.getcwd()+"/"+fileName
        seq_class_prob= torch.softmax(self.seq_weight, dim=-1)
        _, indices = torch.topk(seq_class_prob, 1, -1)
        class_dict= {}
        for i in range(len(indices)):
            class_dict.update({str(i):str(indices[i][0].item())})
        with open(savePath,'w') as fout:
            for key in self.seq_dict.keys():
                seq_index = self.seq_dict[key]
                seq_class = class_dict[str(seq_index)]
                outline = key+"\tclass:"+seq_class+"\n"
                fout.write(outline)

    def saveSeqProb(self,infomessage,global_w,t_w):
        dir = os.getcwd()
        fileName = 'seq_prob_dict.txt'
        savePath = dir + "/" + fileName
        fout = open(savePath, 'a')
        global_w_soft = torch.nn.functional.softmax(global_w/t_w, dim=1)
        print(infomessage,file=fout)
        for key in self.seq_dict.keys():
            seq_index = torch.tensor(int(self.seq_dict[key]))
            seq_index = seq_index.to(self.device).unsqueeze(-1)
            seq_prob = torch.softmax(self.seq_weight(seq_index).squeeze(1), dim = -1)
            print("seq: ",key,file=fout)
            print("prob: ",seq_prob,file=fout)
            print("weight: ",torch.sum(seq_prob*global_w_soft),file=fout)
            print("\n",file=fout)
        print("global_w:",global_w,file=fout)
        print("global_w_soft:",global_w_soft,file=fout)
        fout.close()

class HDRO_learning_w_kuairand(nn.Module):
    def __init__(self, baseModel,
                 num_scenarios,
                 num_cluster_class,
                 n_items,
                 dense_feas,
                 sparse_feas,
                 domain_num,
                 seq_num,
                 seq_dict,
                 alpha,
                 beta,
                 lambda_w,
                 batch_size,
                 pre_dis
                 ):
        super(HDRO_learning_w_kuairand, self).__init__()
        self.num_scenarios = num_scenarios
        self.seq_class = num_cluster_class
        self.n_items = n_items
        self.features = dense_feas + sparse_feas
        self.seq_num = seq_num
        self.seq_dict = seq_dict
        self.alpha = alpha
        self.beta = beta
        self.lambda_w = lambda_w
        self.pre_dis = pre_dis
        self.batch_size=batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.V = (torch.nn.Parameter(data=torch.rand(1, num_cluster_class), requires_grad=True))
        self.V.data /= self.V.data.sum()
        #self.V_pre = [random.uniform(0, 1) for _ in range(num_cluster_class)]
        self.V_pre = np.full(num_cluster_class, 1 / num_cluster_class).tolist()
        self.V_pre_2 = self.observe_distribution_v(self.seq_class)
        # seq_class=60
        # self.V_pre_2 =[0.01645, 0.01649, 0.01775, 0.01772, 0.01862, 0.01454, 0.01426, 0.01574, 0.01574, 0.01513, 0.02041, 0.01853, 0.0171, 0.01669, 0.01431, 0.01757, 0.01746, 0.01695, 0.01731, 0.0156, 0.01852, 0.01408, 0.01994, 0.01828, 0.01794, 0.01753, 0.01872, 0.01538, 0.01458, 0.01677, 0.01976, 0.01581, 0.01616, 0.01815, 0.01592, 0.01473, 0.01656, 0.01626, 0.01496, 0.01838, 0.01857, 0.01803, 0.01752, 0.01521, 0.01843, 0.01668, 0.0173, 0.01588, 0.01887, 0.01616, 0.01491, 0.01502, 0.01421, 0.01495, 0.01474, 0.01594, 0.01381, 0.0147, 0.01774, 0.01854]
        # self.V = (torch.nn.Parameter(torch.full((num_seq,), (num_seq - 1) / 2.0 + 1), requires_grad=True))
        # target_distribution1 = np.array(
        #     [1066194 / (1066194 + 3286302 + 183556 + 402534), 3286302 / (1066194 + 3286302 + 183556 + 402534),
        #      183556 / (1066194 + 3286302 + 183556 + 402534), 402534 / (1066194 + 3286302 + 183556 + 402534)])
        self.t_v = 1
        self.W = (torch.nn.Parameter(data=torch.rand(1, num_scenarios), requires_grad=True))  # 3
        self.W.data /= self.W.data.sum()
        self.W_pre = np.full(num_scenarios, 1/num_scenarios).tolist()
        self.W_pre_2 = [0.21589,0.66543,0.03717,0.08151]
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.criterion = torch.nn.BCELoss()  # default loss cross_entropy
        self.criterion_v = torch.nn.BCELoss(reduction='none')
        self.checkpoint = None
        if baseModel == 'mlp':
            self.baseModel = Mlp_2_Layer(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == 'mlp_adp':
            self.baseModel = MLP_adap_2_layer_1_adp(dense_feas + sparse_feas, domain_num=domain_num,
                                                    fcn_dims=[256, 128],
                                                    hyper_dims=[64], k=35)
        elif baseModel == "dcn_md":
            self.baseModel = DCN_MD(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=2,
                                    mlp_params={"dims": [256, 128]})
        elif baseModel == "dcn_md_adp":
            self.baseModel = DCN_MD_adp(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=2,
                                        k=30, mlp_params={"dims": [256, 128]}, hyper_dims=[128])
        elif baseModel == "wd_md":
            self.baseModel = WideDeep_MD(wide_features=dense_feas, num_domains=domain_num, deep_features=sparse_feas,
                                         mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
        elif baseModel == "wd_md_adp":
            self.baseModel = WideDeep_MD_adp(wide_features=dense_feas, num_domains=domain_num,
                                             deep_features=sparse_feas, k=45,
                                             mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
                                             hyper_dims=[128])
        elif baseModel == "SharedBottom":
            self.baseModel = Mlp_2_Layer_SharedBottom_kuairand(dense_feas + sparse_feas, domain_num=domain_num,
                                                      fcn_dims=[256, 128])
        elif baseModel == "MMoE":
            self.baseModel = Mlp_2_Layer_MMoE_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "PLE":
            self.baseModel = Mlp_2_Layer_PLE_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "AITM":
            self.baseModel = Mlp_2_Layer_AITM_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif baseModel == "STAR":
            self.baseModel = Mlp_2_Layer_STAR_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])

        else:
            pass

        self.emb = self.baseModel.get_emb_dict()

        self.cluster = Encoder_Decoder_kuairand(
            k_class=self.seq_class,
            seq_num=self.seq_num,
            seq_dict=self.seq_dict,
            seq_emb_dim=1100
        )
    def observe_distribution_v(self, cluster_class):
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        data_path = '/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/kuairand4'
        df_train = pd.read_csv(data_path + '/train_dataset.csv')
        # df_val = pd.read_csv(data_path + '/val_dataset.csv')
        # df_test = pd.read_csv(data_path + '/test_dataset_0.8_0.8.csv')
        del df_train["label"]
        # del df_val["label"]
        # del df_test["label"]
        # df_train = df_train[:100]
        # df_val = df_val[:100]
        # df_test = df_test[:100]
        data = df_train

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        user_item_numeric = data[numeric_columns]

        # 使用 PCA 进行降维
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(user_item_numeric)

        # 将用户和物品嵌入连接，形成样本表示
        samples_representation = np.hstack((embedding[:, 0].reshape(-1, 1), embedding[:, 1].reshape(-1, 1)))

        data = samples_representation
        # 使用 K-Means 聚类将数据分成3类
        n_clusters = cluster_class
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)

        # 统计每个类别的样本个数
        counts_by_cluster = np.bincount(labels)

        # 计算正则化后的类别分布
        normalized_distribution = np.array(np.round(normalize(counts_by_cluster.reshape(1, -1), norm='l1'),5)).tolist()[0]

        return normalized_distribution
    def kl_divergence(self,p, q, epsilon=1e-8):
        p = p.cpu()
        p = torch.clamp(p, epsilon, 1 - epsilon)
        q = torch.clamp(q, epsilon, 1 - epsilon)
        return torch.sum(p * torch.log(p / q))

    def w_weight_loss_kuairand(self,x_dict,unweighted_loss,inter_scores_scenario_1,inter_scores_scenario_2,inter_scores_scenario_3,inter_scores_scenario_4):
        len_1= len(inter_scores_scenario_1)
        len_2 = len(inter_scores_scenario_2)
        len_3 = len(inter_scores_scenario_3)
        len_4 = len(inter_scores_scenario_4)
        if len_1 ==0:
            #new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            new_tensor_scenario_4 = unweighted_loss[torch.tensor(inter_scores_scenario_4)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            #new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]
            new_seq_weight_scenario_4 = seq_weight[torch.tensor(inter_scores_scenario_4)]

            #  weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_4 = torch.mean(new_seq_weight_scenario_4 * new_tensor_scenario_4)
            weighted_loss_v_1 = weighted_loss_v_3.detach().clone().fill_(0.0000001).requires_grad_()
        elif len_2 ==0:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            #new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            new_tensor_scenario_4 = unweighted_loss[torch.tensor(inter_scores_scenario_4)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            #new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]
            new_seq_weight_scenario_4 = seq_weight[torch.tensor(inter_scores_scenario_4)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            #weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_4 = torch.mean(new_seq_weight_scenario_4 * new_tensor_scenario_4)
            weighted_loss_v_2 = weighted_loss_v_3.detach().clone().fill_(0.0000001).requires_grad_()
        elif len_3 == 0:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            #new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            new_tensor_scenario_4 = unweighted_loss[torch.tensor(inter_scores_scenario_4)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            #new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]
            new_seq_weight_scenario_4 = seq_weight[torch.tensor(inter_scores_scenario_4)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            # weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_4 = torch.mean(new_seq_weight_scenario_4 * new_tensor_scenario_4)
            weighted_loss_v_3 = weighted_loss_v_2.detach().clone().fill_(0.0000001).requires_grad_()
        elif len_4 == 0:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            #new_tensor_scenario_4 = unweighted_loss[torch.tensor(inter_scores_scenario_4)]
            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]
            #new_seq_weight_scenario_4 = seq_weight[torch.tensor(inter_scores_scenario_4)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            #weighted_loss_v_4 = torch.mean(new_seq_weight_scenario_4 * new_tensor_scenario_4)
            weighted_loss_v_4 = weighted_loss_v_2.detach().clone().fill_(0.0000001).requires_grad_()
        else:
            new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
            new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
            new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
            new_tensor_scenario_4 = unweighted_loss[torch.tensor(inter_scores_scenario_4)]

            v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
            seq_class_softmax = self.cluster.get_probability(x_dict)
            seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
            new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
            new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
            new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]
            new_seq_weight_scenario_4 = seq_weight[torch.tensor(inter_scores_scenario_4)]

            weighted_loss_v_1 = torch.mean(new_seq_weight_scenario_1 * new_tensor_scenario_1)
            weighted_loss_v_2 = torch.mean(new_seq_weight_scenario_2 * new_tensor_scenario_2)
            weighted_loss_v_3 = torch.mean(new_seq_weight_scenario_3 * new_tensor_scenario_3)
            weighted_loss_v_4 = torch.mean(new_seq_weight_scenario_4 * new_tensor_scenario_4)

        return weighted_loss_v_1,weighted_loss_v_2,weighted_loss_v_3,weighted_loss_v_4

    def forward(self, x_dict, y):
        self.emb = self.baseModel.get_emb_dict()
        embs4cluster = self.emb(x_dict, self.features)
        cluster_loss = self.cluster(x_dict, embs4cluster)
        scores = self.baseModel(x_dict)

        #  add scenarios weight V W in training process
        unweighted_loss = self.criterion_v(scores.float(), y.float())

        scenarios_ids = [0,1,2,3]

        inter_scores_scenario_1 = []
        inter_scores_scenario_2 = []
        inter_scores_scenario_3 = []
        inter_scores_scenario_4 = []

        x_dict_list = x_dict["domain_indicator"].tolist()
        len_indicator = len(x_dict_list)

        for i in range(len_indicator):
            if x_dict_list[i] == scenarios_ids[0]:
                inter_scores_scenario_1.append(i)
            elif x_dict_list[i] == scenarios_ids[1]:
                inter_scores_scenario_2.append(i)
            elif x_dict_list[i] == scenarios_ids[2]:
                inter_scores_scenario_3.append(i)
            elif x_dict_list[i] == scenarios_ids[3]:
                inter_scores_scenario_4.append(i)

        if len(inter_scores_scenario_1) ==0:
            inter_scores_scenario_1= [0, 2]
        else:
            inter_scores_scenario_1= inter_scores_scenario_1

        if len(inter_scores_scenario_2) ==0:
            inter_scores_scenario_2= [1, 2]
        else:
            inter_scores_scenario_2= inter_scores_scenario_2

        if len(inter_scores_scenario_3) ==0:
            inter_scores_scenario_3= [16]
        else:
            inter_scores_scenario_3= inter_scores_scenario_3

        if len(inter_scores_scenario_4) ==0:
            inter_scores_scenario_4= [1]
        else:
            inter_scores_scenario_4=inter_scores_scenario_4

        # len_score_1 = len(inter_scores_scenario_1)
        # len_score_2 = len(inter_scores_scenario_2)
        # len_score_3 = len(inter_scores_scenario_3)
        # len_score_4 = len(inter_scores_scenario_4)
        # if len_score_1 == 0 or len_score_2 == 0 or len_score_3 == 0 or len_score_4 == 0 :
        #     weighted_loss_v_1, weighted_loss_v_2, weighted_loss_v_3, weighted_loss_v_4 = self.w_weight_loss_kuairand(x_dict, unweighted_loss,
        #                                                                                  inter_scores_scenario_1,
        #                                                                                  inter_scores_scenario_2,
        #                                                                                  inter_scores_scenario_3,
        #                                                                                  inter_scores_scenario_4)
        # else:
        new_tensor_scenario_1 = unweighted_loss[torch.tensor(inter_scores_scenario_1)]
        new_tensor_scenario_2 = unweighted_loss[torch.tensor(inter_scores_scenario_2)]
        new_tensor_scenario_3 = unweighted_loss[torch.tensor(inter_scores_scenario_3)]
        new_tensor_scenario_4 = unweighted_loss[torch.tensor(inter_scores_scenario_4)]
        v_softmax = self.V
        # v_softmax = torch.nn.functional.softmax(self.V / self.t_v, dim=1)
        seq_class_softmax = self.cluster.get_probability(x_dict)
        seq_weight = torch.matmul(seq_class_softmax, v_softmax.T)
        new_seq_weight_scenario_1 = seq_weight[torch.tensor(inter_scores_scenario_1)]
        new_seq_weight_scenario_2 = seq_weight[torch.tensor(inter_scores_scenario_2)]
        new_seq_weight_scenario_3 = seq_weight[torch.tensor(inter_scores_scenario_3)]
        new_seq_weight_scenario_4 = seq_weight[torch.tensor(inter_scores_scenario_4)]

        weighted_loss_v_1 = torch.sum(new_seq_weight_scenario_1 * new_tensor_scenario_1)
        weighted_loss_v_2 = torch.sum(new_seq_weight_scenario_2 * new_tensor_scenario_2)
        weighted_loss_v_3 = torch.sum(new_seq_weight_scenario_3 * new_tensor_scenario_3)
        weighted_loss_v_4 = torch.sum(new_seq_weight_scenario_4 * new_tensor_scenario_4)
        #
        # weighted_loss_v_1, weighted_loss_v_2, weighted_loss_v_3, weighted_loss_v_4 = self.w_weight_loss_kuairand(x_dict,unweighted_loss,inter_scores_scenario_1,
        #                                                                                                          inter_scores_scenario_2,inter_scores_scenario_3,
        #                                                                                                         inter_scores_scenario_4)

        #weighted_loss_v_loss = torch.mean(weighted_loss_v_1 + weighted_loss_v_2 + weighted_loss_v_3+ weighted_loss_v_4)
        weighted_loss_w_distribution = torch.stack([weighted_loss_v_1, weighted_loss_v_2, weighted_loss_v_3, weighted_loss_v_4])
        weighted_loss_w_loss = torch.sum(self.W * weighted_loss_w_distribution)
        # weighted_loss = weighted_loss_v_loss
        weighted_loss = weighted_loss_w_loss

            # kl V loss
        if self.pre_dis == 1:
            with torch.no_grad():
                self.V_pre = torch.as_tensor(self.V_pre).clone()
            kl_V_loss = self.kl_divergence(self.V, self.V_pre)
            weighted_loss = weighted_loss + self.alpha * kl_V_loss

            # kl w loss
            with torch.no_grad():
                self.W_pre = torch.as_tensor(self.W_pre).clone()
            kl_W_loss = self.kl_divergence(self.W, self.W_pre)
            weighted_loss = weighted_loss + self.lambda_w * kl_W_loss
        else:
            with torch.no_grad():
                self.V_pre_2 = torch.as_tensor(self.V_pre_2).clone()
            kl_V_loss = self.kl_divergence(self.V, self.V_pre_2)
            weighted_loss = weighted_loss + self.alpha * kl_V_loss

            # kl w loss
            with torch.no_grad():
                self.W_pre_2 = torch.as_tensor(self.W_pre_2).clone()
            kl_W_loss = self.kl_divergence(self.W, self.W_pre_2)
            weighted_loss = weighted_loss + self.lambda_w * kl_W_loss

        return weighted_loss, self.beta * cluster_loss