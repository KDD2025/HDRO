import sys
sys.path.append("../")
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from MSRS.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from MSRS.trainers import CTRTrainer
from MSRS.trainers.ctr_trainer import CTRTrainerV
from MSRS.utils.data import DataGenerator
from MSRS.models.multi_domain import Mlp_7_Layer, Mlp_2_Layer, MLP_adap_2_layer_1_adp, DCN_MD, DCN_MD_adp, WideDeep_MD, WideDeep_MD_adp
from MSRS.models.multi_domain.adapter import Mlp_2_Layer_SharedBottom,Mlp_2_Layer_MMoE,Mlp_2_Layer_PLE,Mlp_2_Layer_AITM,Mlp_2_Layer_STAR
from MSRS.models.multi_domain.adapter import Mlp_2_Layer_SharedBottom_kuairand,Mlp_2_Layer_MMoE_kuairand,Mlp_2_Layer_PLE_kuairand,Mlp_2_Layer_AITM_kuairand,Mlp_2_Layer_STAR_kuairand
from MSRS.Utility.tools import validate,getSeqDict,saveModel,readSeqDict,seq2str,getSeqDict_kuairand,getSeqDict_kuairand4
from MSRS.models.HDRO import HDRO_learning_w,HDRO_learning_w_kuairand

# data cluster and domain_ratio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import minimize

def kl_divergence(p, q, epsilon=1e-8):
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)
    return np.sum(p * np.log(p / q))

def kl_constraint(x, target_distribution, kl_target):
    return kl_divergence(target_distribution, x) - kl_target

def find_distribution_with_kl(target_distribution, kl_target):
    n = len(target_distribution)
    initial_guess = np.ones(n) / n
    objective_function = lambda x: np.sum((target_distribution - x) ** 2)  # 这里使用平方和作为目标函数

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: kl_constraint(x, target_distribution, kl_target)})

    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)
    return result.x
def cluster_2(data):
    # 提取数值型的列
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    user_item_numeric = data[numeric_columns]

    # 使用 PCA 进行降维
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(user_item_numeric)

    # 将用户和物品嵌入连接，形成样本表示
    samples_representation = np.hstack((embedding[:, 0].reshape(-1, 1), embedding[:, 1].reshape(-1, 1)))

    # 使用 k-means 进行聚类，这里假设分为 2 类
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(samples_representation)

    # 获取聚类结果
    labels = kmeans.labels_

    # 将聚类结果添加到原始 DataFrame 中
    data['label'] = labels

    return data

def get_kuairand_data_rank_multidomain(data_path="/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/kuairand4"):
    df_train = pd.read_csv(data_path + '/train_dataset.csv')
    df_val = pd.read_csv(data_path + '/val_dataset.csv')
    df_test = pd.read_csv(data_path + '/test_dataset_0.8_0.8.csv')
    del df_train["label"]
    del df_val["label"]
    del df_test["label"]
    # df_train=df_train[:50000]
    # df_val = df_val[:50000]
    # df_test = df_test[:50000]
    #print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)

    n_items = len(set(data["item_id"].tolist()))
    domain_num = 4

    col_names = data.columns.values.tolist()
    dense_cols = ['tab']
    sparse_cols = ['user_id', 'item_id', "domain_indicator"]
    #print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]

    y = data["target"]
    del data["target"]
    x = data
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:], y[val_idx:]
    return dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num,n_items



def map_group_indicator(tab, list_group):
    l = len(list(list_group))
    for i in range(l):
        if tab in list_group[i]:
            return i


def convert_target(val):
    v = int(val)
    if v > 3:
        return int(1)
    else:
        return int(0)

def convert_target_kuairand(val):
    v = int(val)
    if v == 1:
        return int(1)
    else:
        return int(0)

def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)

def df_to_dict_multi_domain(data, columns):
    """
    Convert the array to a dict type input that the network can accept
    Args:
        data (array): 3D datasets of type DataFrame (Length * Domain_num * feature_num)
        columns (list): feature name list
    Returns:
        The converted dict, which can be used directly into the input network
    """

    data_dict = {}
    for i in range(len(columns)):
        data_dict[columns[i]] = data[:, :, i]
    return data_dict

def main(model_name_type,dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num,n_items = get_kuairand_data_rank_multidomain(
        dataset_path)
    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test,
                                                                               y_test=y_test, batch_size=batch_size)

    import os
    torch.cuda.set_device(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name_type == "Base":
        if model_name == "mlp":
            model = Mlp_2_Layer(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif model_name == "mlp_adp":
            model = MLP_adap_2_layer_1_adp(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128],
                                           hyper_dims=[64], k=35)
        elif model_name == "dcn_md":
            model = DCN_MD(features=dense_feas + sparse_feas,num_domains=domain_num ,n_cross_layers=2, mlp_params={"dims": [256, 128]})
        elif model_name == "dcn_md_adp":
            model = DCN_MD_adp(features=dense_feas + sparse_feas,num_domains=domain_num, n_cross_layers=2, k = 30, mlp_params={"dims": [256, 128]}, hyper_dims=[128])
        elif model_name == "wd_md":
            model = WideDeep_MD(wide_features=dense_feas,num_domains= domain_num, deep_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
        elif model_name == "wd_md_adp":
            model = WideDeep_MD_adp(wide_features=dense_feas,num_domains= domain_num, deep_features=sparse_feas,  k= 45,mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}, hyper_dims=[128])
        elif model_name == "SharedBottom":
            model = Mlp_2_Layer_SharedBottom_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif model_name == "MMoE":
            model = Mlp_2_Layer_MMoE_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif model_name == "PLE":
            model = Mlp_2_Layer_PLE_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif model_name == "AITM":
            model = Mlp_2_Layer_AITM_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
        elif model_name == "STAR":
            model = Mlp_2_Layer_STAR_kuairand(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])

        ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=10, device=device, model_path=save_dir,scheduler_params={"step_size": 4,"gamma": 0.85})

        ctr_trainer.fit(train_dataloader, val_dataloader)

        # auc1, auc2, auc3, auc = ctr_trainer.evaluate_multi_domain_auc(ctr_trainer.model, test_dataloader)
        # log1, log2, log3, log = ctr_trainer.evaluate_multi_domain_logloss(ctr_trainer.model, test_dataloader)
        auc1, auc2, auc3,auc4, auc = ctr_trainer.evaluate_multi_domain_auc_kuairand4(ctr_trainer.model, test_dataloader)
        log1, log2, log3,log4,log = ctr_trainer.evaluate_multi_domain_logloss_kuairand4(ctr_trainer.model, test_dataloader)
        print("=================================================================================================")
        print(f'test auc: {auc} | test logloss: {log}')
        print(f'domain 1 test auc: {auc1} | test logloss: {log1}')
        print(f'domain 2 test auc: {auc2} | test logloss: {log2}')
        print(f'domain 3 test auc: {auc3} | test logloss: {log3}')
        print(f'domain 4 test auc: {auc4} | test logloss: {log4}')

        # save csv file
        from datetime import datetime
        import os
        import csv
        current_time = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        csv_file_path = os.path.join(args.result_path,
                                     f"{args.model_name_type}_{args.data_name}_{model_name}_{seed}_{current_time}.csv")
        # with open(model_name+"_"+str(seed)+"_"+str(current_time)+'.csv', "w", newline='') as f:
        with open(csv_file_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'seed', 'auc', 'log', 'auc1', 'log1', 'auc2', 'log2', 'auc3', 'log3', 'auc4', 'log4'])
            writer.writerow([model_name, str(seed), auc, log, auc1, log1, auc2, log2, auc3, log3, auc4, log4])
    # add Weight V IN inter-level
    elif model_name_type == "HDRO":
        seq_num, seq_dict = getSeqDict_kuairand4(train_dataloader)
        model = HDRO_learning_w_kuairand(baseModel=model_name,
            num_scenarios = domain_num,
            num_cluster_class=args.cluster_class,
            n_items=n_items,
            dense_feas =dense_feas,
            sparse_feas=sparse_feas,
            domain_num = domain_num,
            seq_num=seq_num,
            seq_dict=seq_dict,
            alpha =args.balance_v_loss,
            beta=args.balance_cluster_loss,
            lambda_w=args.balance_w_loss,
            batch_size=batch_size,
            pre_dis=args.pre_distribution
        ).to(device)
        ctr_trainer = CTRTrainerV(model,
                                 args.learning_rate_f,
                                  args.learning_rate_v,
                                  args.learning_rate_w,
                                  args.learning_rate_c,
                                  n_epoch=epoch,
                                  earlystop_patience=10,
                                  device=device,
                                  model_path=save_dir,
                                  scheduler_params={"step_size": 4, "gamma": 0.85})

        ctr_trainer.fit(train_dataloader, val_dataloader)
        auc1, auc2, auc3, auc4, auc = ctr_trainer.evaluate_multi_domain_auc_kuairand4(ctr_trainer.model,test_dataloader)
        log1, log2, log3, log4, log = ctr_trainer.evaluate_multi_domain_logloss_kuairand4(ctr_trainer.model,test_dataloader)
        # print("=================================================================================================")
        # print(f'test auc: {auc} | test logloss: {log}')
        print(f'domain 1 | test auc: {auc1} | test logloss: {log1}')
        print(f'domain 2 | test auc: {auc2} | test logloss: {log2}')
        print(f'domain 3 | test auc: {auc3} | test logloss: {log3}')
        print(f'domain 4 | test auc: {auc4} | test logloss: {log4}')

        # save csv file
        from datetime import datetime
        import os
        import csv
        current_time = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        csv_file_path = os.path.join(args.result_path,
                                     f"{args.model_name_type}_{args.data_name}_{model_name}_{seed}_{current_time}.csv")
        # with open(model_name+"_"+str(seed)+"_"+str(current_time)+'.csv', "w", newline='') as f:
        with open(csv_file_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['model', 'seed', 'auc', 'log', 'auc1', 'log1', 'auc2', 'log2', 'auc3', 'log3', 'auc4', 'log4'])
            writer.writerow(
                [model_name, str(seed), auc, log, auc1, log1, auc2, log2, auc3, log3, auc4, log4])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        default="/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/kuairand4")  # /home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/kuairand4/
    parser.add_argument('--model_name', default='mlp_adp')
    parser.add_argument('--epoch', type=int, default=100)  # 100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096 * 10)  # 4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')  # cuda:0 default='cpu'
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    # parser.add_argument("--gpu_id", type=str, default="1", help="gpu_id")
    parser.add_argument("--gpu_id", type=int, default=1, help="gpu_id")
    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument('--model_name_type', type=str, default='Base', help='model types of Learn Weight')
    parser.add_argument('--result_path', default="./Test-Result")
    parser.add_argument('--data_name', default="kuairand")
    parser.add_argument('--cluster_class', type=int, default=10)
    parser.add_argument('--pre_distribution', type=int, default=1)
    parser.add_argument('--learning_rate_v', type=float, default=1e-8)
    parser.add_argument('--learning_rate_w', type=float, default=1e-8)
    parser.add_argument('--learning_rate_c', type=float, default=1e-8)
    parser.add_argument('--learning_rate_f', type=float, default=1e-8)
    parser.add_argument('--num_scenarios', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--balance_v_loss', '-alpha', type=float, default=0.6,
                        help='balance of interaction level KL loss')
    parser.add_argument('--balance_cluster_loss', '-beta', type=float, default=0.6,
                        help='balance of interaction cluster loss')
    parser.add_argument('--balance_w_loss', '-lambda_w', type=float, default=0.6,
                        help='balance of scenarios level KL loss')
    parser.add_argument('--dis_scenario', '-tau1', type=float, default=0.8,
                        help='balance_data_distribution_scenario in multi domains')
    parser.add_argument('--dis_interaction', '-tau2', type=float, default=0.8,
                        help='balance_data_distribution_scenario in multi domains')
    parser.add_argument('--expert_num', type=int, default=4)

    args = parser.parse_args()
    main(args.model_name_type, args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size,
         args.weight_decay, args.device, args.save_dir, args.seed)
# nohup python run_kuairand_train_val_test_multi_scenario.py --data_name kuairand4 --model_name_type HDRO --model_name STAR --epoch 200
# --batch_size=512 --seed 2022 --gpu_id 2 --learning_rate_v=1e-2 --learning_rate_w=1e-3 --learning_rate_c=1e-3 --learning_rate_f=1e-3 -alpha=0.01 -beta=0.01 -lambda_w=0.01 --cluster_class=60 --pre_distribution=2
# > Output/kuairand/Kuairand-HDRO-STAR-1-24-gpu2-cluster60-multi-scenario-rl1e-3-alpha-beta-lamda0.01-predis2.log 2>&1 &
