import random
import torch
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split

class TorchDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class PredictDataset(Dataset):

    def __init__(self, x):
        super().__init__()
        self.x = x

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])


class MatchDataGenerator(object):

    def __init__(self, x, y=[]):
        super().__init__()
        if len(y) != 0:
            self.dataset = TorchDataset(x, y)
        else:  # For pair-wise model, trained without given label
            self.dataset = PredictDataset(x)

    def generate_dataloader(self, x_test_user, x_all_item, batch_size, num_workers=8):
        train_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = PredictDataset(x_test_user)

        # shuffle = False to keep same order as ground truth
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        item_dataset = PredictDataset(x_all_item)
        item_dataloader = DataLoader(item_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, test_dataloader, item_dataloader


class DataGenerator(object):

    def __init__(self, x, y):
        super().__init__()
        self.dataset = TorchDataset(x, y)
        self.length = len(self.dataset)

    def condition_test_sample_domain1_label0(self,sample):

        return sample[0]['domain_indicator'] == 0 and sample[0]['label'] == 0

    def condition_test_sample_domain1_label1(self,sample):

        return sample[0]['domain_indicator'] == 0 and sample[0]['label'] == 1

    def condition_test_sample_domain2_label0(self,sample):

        return sample[0]['domain_indicator'] == 1 and sample[0]['label'] == 0

    def condition_test_sample_domain2_label1(self,sample):

        return sample[0]['domain_indicator'] == 1 and sample[0]['label'] == 1

    def condition_test_sample_domain3_label0(self, sample):

        return sample[0]['domain_indicator'] == 2 and sample[0]['label'] == 0

    def condition_test_sample_domain3_label1(self, sample):

        return sample[0]['domain_indicator'] == 2 and sample[0]['label'] == 1

    def condition_test_sample_domain4_label0(self, sample):

        return sample[0]['domain_indicator'] == 3 and sample[0]['label'] == 0

    def condition_test_sample_domain4_label1(self, sample):

        return sample[0]['domain_indicator'] == 3 and sample[0]['label'] == 1
    def sample_with_original_ratio(self,a_samples, b_samples, total_samples):
        size_a = len(a_samples)
        size_b = len(b_samples)
        # 计算原始样本集A和B的比例
        ratio_a = size_a / (size_a + size_b)

        ratio_b = size_b / (size_a + size_b)
        #print("ratio", ratio_a,ratio_b)
        # 计算在m个样本中应该有多少个样本属于A和B
        count_a = int(ratio_a * total_samples)
        count_b = total_samples - count_a
        # 从样本集A中随机抽取count_a个样本
        selected_a_samples = random.sample(a_samples, min(count_a, size_a))
        # 从样本集B中随机抽取count_b个样本
        selected_b_samples = random.sample(b_samples, min(count_b, size_b))

        # 返回抽取的样本
        return selected_a_samples + selected_b_samples

    def domain_three(self,test_dataset,domain_ratio,tau2):
        domain1_label0_test = 0
        domain1_label1_test = 0
        domain2_label0_test = 0
        domain2_label1_test = 0
        domain3_label0_test = 0
        domain3_label1_test = 0

        for sample in test_dataset:
            if sample[0]['domain_indicator'] == 0:
                if sample[0]['label'] == 0:
                    domain1_label0_test += 1
                else:
                    domain1_label1_test += 1
            elif sample[0]['domain_indicator'] == 1:
                if sample[0]['label'] == 0:
                    domain2_label0_test += 1
                else:
                    domain2_label1_test += 1
            else:
                if sample[0]['label'] == 0:
                    domain3_label0_test += 1
                else:
                    domain3_label1_test += 1

        test_dataset_domain1_label0 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain1_label0(sample)]
        test_dataset_domain1_label1 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain1_label1(sample)]
        test_dataset_domain2_label0 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain2_label0(sample)]
        test_dataset_domain2_label1 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain2_label1(sample)]
        test_dataset_domain3_label0 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain3_label0(sample)]
        test_dataset_domain3_label1 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain3_label1(sample)]

        beta = tau2

        if beta == 1:
            test_tau_domain_1 = [0, domain1_label1_test]
            test_tau_domain_2 = [0, domain2_label1_test]
            test_tau_domain_3 = [0, domain3_label1_test]
        else:
            if domain1_label0_test < domain1_label0_test:
                test_tau_domain_1 = [int((1 - beta) * domain1_label0_test), domain1_label1_test]
            else:
                test_tau_domain_1 = [domain1_label0_test, int((1 - beta) * domain1_label1_test)]

            if domain2_label0_test < domain2_label0_test:
                test_tau_domain_2 = [int((1 - beta) * domain2_label0_test), domain2_label1_test]
            else:
                test_tau_domain_2 = [domain2_label0_test, int((1 - beta) * domain2_label1_test)]

            if domain3_label0_test < domain3_label0_test:
                test_tau_domain_3 = [int((1 - beta) * domain3_label0_test), domain3_label1_test]
            else:
                test_tau_domain_3 = [domain3_label0_test, int((1 - beta) * domain3_label1_test)]

        if len(test_dataset_domain1_label0) < test_tau_domain_1[0]:
            test_dataset_domain1_label0_sample = test_dataset_domain1_label0
        else:
            test_dataset_domain1_label0_sample = random.sample(test_dataset_domain1_label0, test_tau_domain_1[0])
        if len(test_dataset_domain1_label1) < test_tau_domain_1[1]:
            test_dataset_domain1_label1_sample = test_dataset_domain1_label1
        else:
            test_dataset_domain1_label1_sample = random.sample(test_dataset_domain1_label1, test_tau_domain_1[1])
        test_dataset_domain1_label_len = test_tau_domain_1[0] + test_tau_domain_1[1]

        if len(test_dataset_domain2_label0) < test_tau_domain_2[0]:
            test_dataset_domain2_label0_sample = test_dataset_domain2_label0
        else:
            test_dataset_domain2_label0_sample = random.sample(test_dataset_domain2_label0, test_tau_domain_2[0])
        if len(test_dataset_domain2_label1) < test_tau_domain_2[1]:
            test_dataset_domain2_label1_sample = test_dataset_domain2_label1
        else:
            test_dataset_domain2_label1_sample = random.sample(test_dataset_domain2_label1, test_tau_domain_2[1])
        test_dataset_domain2_label_len = test_tau_domain_2[0] + test_tau_domain_2[1]

        if len(test_dataset_domain3_label0) < test_tau_domain_3[0]:
            test_dataset_domain3_label0_sample = test_dataset_domain3_label0
        else:
            test_dataset_domain3_label0_sample = random.sample(test_dataset_domain3_label0, test_tau_domain_3[0])
        if len(test_dataset_domain3_label1) < test_tau_domain_3[1]:
            test_dataset_domain3_label1_sample = test_dataset_domain3_label1
        else:
            test_dataset_domain3_label1_sample = random.sample(test_dataset_domain3_label1, test_tau_domain_3[1])
        test_dataset_domain3_label_len = test_tau_domain_3[0] + test_tau_domain_3[1]

        test_dataset_domain_len = test_dataset_domain1_label_len + test_dataset_domain2_label_len + test_dataset_domain3_label_len

        test_dataset_domain1_sample_num = int(domain_ratio[0] * test_dataset_domain_len)
        test_dataset_domain2_sample_num = int(domain_ratio[1] * test_dataset_domain_len)
        test_dataset_domain3_sample_num = test_dataset_domain_len - test_dataset_domain1_sample_num - test_dataset_domain2_sample_num

        # 在每个样本集中按照计算出的样本个数取样本
        selected_samples_a = self.sample_with_original_ratio(test_dataset_domain1_label0_sample,
                                                             test_dataset_domain1_label1_sample,
                                                             test_dataset_domain1_sample_num)
        selected_samples_b = self.sample_with_original_ratio(test_dataset_domain2_label0_sample,
                                                             test_dataset_domain2_label1_sample,
                                                             test_dataset_domain2_sample_num)
        selected_samples_c = self.sample_with_original_ratio(test_dataset_domain3_label0_sample,
                                                             test_dataset_domain3_label1_sample,
                                                             test_dataset_domain3_sample_num)
        test_dataset = selected_samples_a + selected_samples_b + selected_samples_c
        return  test_dataset

    def domain_four(self,test_dataset,domain_ratio,tau2):
        domain1_label0_test = 0
        domain1_label1_test = 0
        domain2_label0_test = 0
        domain2_label1_test = 0
        domain3_label0_test = 0
        domain3_label1_test = 0
        domain4_label0_test = 0
        domain4_label1_test = 0

        for sample in test_dataset:
            if sample[0]['domain_indicator'] == 0:
                if sample[0]['label'] == 0:
                    domain1_label0_test += 1
                else:
                    domain1_label1_test += 1
            elif sample[0]['domain_indicator'] == 1:
                if sample[0]['label'] == 0:
                    domain2_label0_test += 1
                else:
                    domain2_label1_test += 1
            elif sample[0]['domain_indicator'] == 2:
                if sample[0]['label'] == 0:
                    domain3_label0_test += 1
                else:
                    domain3_label1_test += 1
            else:
                if sample[0]['label'] == 0:
                    domain4_label0_test += 1
                else:
                    domain4_label1_test += 1

        test_dataset_domain1_label0 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain1_label0(sample)]
        test_dataset_domain1_label1 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain1_label1(sample)]
        test_dataset_domain2_label0 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain2_label0(sample)]
        test_dataset_domain2_label1 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain2_label1(sample)]
        test_dataset_domain3_label0 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain3_label0(sample)]
        test_dataset_domain3_label1 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain3_label1(sample)]
        test_dataset_domain4_label0 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain4_label0(sample)]
        test_dataset_domain4_label1 = [sample for sample in test_dataset if
                                       self.condition_test_sample_domain4_label1(sample)]

        beta = tau2

        if beta == 1:
            test_tau_domain_1 = [0, domain1_label1_test]
            test_tau_domain_2 = [0, domain2_label1_test]
            test_tau_domain_3 = [0, domain3_label1_test]
            test_tau_domain_4 = [0, domain4_label1_test]
        else:
            if domain1_label0_test < domain1_label0_test:
                test_tau_domain_1 = [int((1 - beta) * domain1_label0_test), domain1_label1_test]
            else:
                test_tau_domain_1 = [domain1_label0_test, int((1 - beta) * domain1_label1_test)]

            if domain2_label0_test < domain2_label0_test:
                test_tau_domain_2 = [int((1 - beta) * domain2_label0_test), domain2_label1_test]
            else:
                test_tau_domain_2 = [domain2_label0_test, int((1 - beta) * domain2_label1_test)]

            if domain3_label0_test < domain3_label0_test:
                test_tau_domain_3 = [int((1 - beta) * domain3_label0_test), domain3_label1_test]
            else:
                test_tau_domain_3 = [domain3_label0_test, int((1 - beta) * domain3_label1_test)]

            if domain4_label0_test < domain4_label0_test:
                test_tau_domain_4 = [int((1 - beta) * domain4_label0_test), domain4_label1_test]
            else:
                test_tau_domain_4 = [domain4_label0_test, int((1 - beta) * domain4_label1_test)]
        if len(test_dataset_domain1_label0) < test_tau_domain_1[0]:
            test_dataset_domain1_label0_sample = test_dataset_domain1_label0
        else:
            test_dataset_domain1_label0_sample = random.sample(test_dataset_domain1_label0, test_tau_domain_1[0])
        if len(test_dataset_domain1_label1) < test_tau_domain_1[1]:
            test_dataset_domain1_label1_sample = test_dataset_domain1_label1
        else:
            test_dataset_domain1_label1_sample = random.sample(test_dataset_domain1_label1, test_tau_domain_1[1])
        test_dataset_domain1_label_len = test_tau_domain_1[0] + test_tau_domain_1[1]

        if len(test_dataset_domain2_label0) < test_tau_domain_2[0]:
            test_dataset_domain2_label0_sample = test_dataset_domain2_label0
        else:
            test_dataset_domain2_label0_sample = random.sample(test_dataset_domain2_label0, test_tau_domain_2[0])
        if len(test_dataset_domain2_label1) < test_tau_domain_2[1]:
            test_dataset_domain2_label1_sample = test_dataset_domain2_label1
        else:
            test_dataset_domain2_label1_sample = random.sample(test_dataset_domain2_label1, test_tau_domain_2[1])
        test_dataset_domain2_label_len = test_tau_domain_2[0] + test_tau_domain_2[1]

        if len(test_dataset_domain3_label0) < test_tau_domain_3[0]:
            test_dataset_domain3_label0_sample = test_dataset_domain3_label0
        else:
            test_dataset_domain3_label0_sample = random.sample(test_dataset_domain3_label0, test_tau_domain_3[0])
        if len(test_dataset_domain3_label1) < test_tau_domain_3[1]:
            test_dataset_domain3_label1_sample = test_dataset_domain3_label1
        else:
            test_dataset_domain3_label1_sample = random.sample(test_dataset_domain3_label1, test_tau_domain_3[1])
        test_dataset_domain3_label_len = test_tau_domain_3[0] + test_tau_domain_3[1]

        if len(test_dataset_domain4_label0) < test_tau_domain_4[0]:
            test_dataset_domain4_label0_sample = test_dataset_domain4_label0
        else:
            test_dataset_domain4_label0_sample = random.sample(test_dataset_domain4_label0, test_tau_domain_4[0])
        if len(test_dataset_domain4_label1) < test_tau_domain_4[1]:
            test_dataset_domain4_label1_sample = test_dataset_domain4_label1
        else:
            test_dataset_domain4_label1_sample = random.sample(test_dataset_domain4_label1, test_tau_domain_4[1])
        test_dataset_domain4_label_len = test_tau_domain_4[0] + test_tau_domain_4[1]

        test_dataset_domain_len = test_dataset_domain1_label_len + test_dataset_domain2_label_len + test_dataset_domain3_label_len+ test_dataset_domain4_label_len

        test_dataset_domain1_sample_num = int(domain_ratio[0] * test_dataset_domain_len)
        test_dataset_domain2_sample_num = int(domain_ratio[1] * test_dataset_domain_len)
        test_dataset_domain3_sample_num = int(domain_ratio[2] * test_dataset_domain_len)
        test_dataset_domain4_sample_num = test_dataset_domain_len - test_dataset_domain1_sample_num - test_dataset_domain2_sample_num- test_dataset_domain3_sample_num

        # 在每个样本集中按照计算出的样本个数取样本
        selected_samples_a = self.sample_with_original_ratio(test_dataset_domain1_label0_sample,
                                                             test_dataset_domain1_label1_sample,
                                                             test_dataset_domain1_sample_num)
        selected_samples_b = self.sample_with_original_ratio(test_dataset_domain2_label0_sample,
                                                             test_dataset_domain2_label1_sample,
                                                             test_dataset_domain2_sample_num)
        selected_samples_c = self.sample_with_original_ratio(test_dataset_domain3_label0_sample,
                                                             test_dataset_domain3_label1_sample,
                                                             test_dataset_domain3_sample_num)
        selected_samples_d = self.sample_with_original_ratio(test_dataset_domain4_label0_sample,
                                                             test_dataset_domain4_label1_sample,
                                                             test_dataset_domain4_sample_num)
        test_dataset = selected_samples_a + selected_samples_b + selected_samples_c+ selected_samples_d
        return  test_dataset
    def generate_dataloader(self, x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16,
                            num_workers=1, drop_last_flag=False,domain_ratio=None,domain_num=3,tau1=0.8,tau=0.8):
        if split_ratio != None:
            train_length = int(self.length * split_ratio[0])
            val_length = int(self.length * split_ratio[1])
            test_length = self.length - train_length - val_length
            print("the samples of train : val : test are  %d : %d : %d" % (train_length, val_length, test_length))
            train_dataset, val_dataset, test_dataset = random_split(self.dataset,(train_length, val_length, test_length))
            #print("test_dataset000", len(test_dataset))
            # for sample in test_dataset:
            #     print("sample",sample)
            #     exit()
            if domain_ratio !=None:
                if domain_num == 3:
                    #print("333333")
                    test_dataset =self.domain_three(test_dataset,domain_ratio,tau)
                else:
                    #print("444444")
                    test_dataset = self.domain_four(test_dataset,domain_ratio,tau)
            # sample_movielens: [{'user_id': sample[0]['user_id'],'movie_id': sample[0]['movie_id'],'gender': sample[0]['gender'],'age': sample[0]['age'],
            #                           'occupation': sample[0]['occupation'],'zip': sample[0]['zip'],'label': sample[0]['label'],'cate_id': sample[0]['cate_id'],
            #                           'domain_indicator': sample[0]['domain_indicator'],'target': sample[1]} for sample in train_dataset]

            # sample_douban:[{'user_id': sample[0]['user_id'],
            #                 'item_id': sample[0]['item_id'],
            #                 'domain_id': sample[0]['domain_id'],
            #                 'label': sample[0]['label'],
            #                 'domain_indicator': sample[0]['domain_indicator'],'target': sample[1]} for sample in train_dataset]

            # sample_kuairand:[{'user_id': sample[0]['user_id'],
            #                 'item_id': sample[0]['item_id'],
            #                 'tab': sample[0]['tab'],
            #                 'label': sample[0]['label'],
            #                 'domain_indicator': sample[0]['domain_indicator'],'target': sample[1]} for sample in train_dataset]

            # data_list_train = [{'user_id': sample[0]['user_id'],
            #                  'item_id': sample[0]['item_id'],
            #                  'tab': sample[0]['tab'],
            #                  'label': sample[0]['label'],
            #                  'domain_indicator': sample[0]['domain_indicator'],'target': sample[1]} for sample in train_dataset]
            # data_list_val = [{'user_id': sample[0]['user_id'],
            #                  'item_id': sample[0]['item_id'],
            #                  'tab': sample[0]['tab'],
            #                  'label': sample[0]['label'],
            #                  'domain_indicator': sample[0]['domain_indicator'],'target': sample[1]} for sample in val_dataset]

            data_list_test = [{'user_id': sample[0]['user_id'],
                            'item_id': sample[0]['item_id'],
                            'domain_id': sample[0]['domain_id'],
                            'label': sample[0]['label'],
                             'domain_indicator': sample[0]['domain_indicator'],'target': sample[1]} for sample in test_dataset]
            # 转换为 DataFrame
            # df_train = pd.DataFrame(data_list_train)
            # df_val = pd.DataFrame(data_list_val)
            df_test = pd.DataFrame(data_list_test)
            print("df_test",len(df_test))
            # 保存 DataFrame 到 CSV 文件
            import os
            # python run_kuairand4_rank_multi_scenario.py --dis_scenario=0.0 --dis_interaction=0.6 //kuairand4
            # nohup python run_movielens_rank_multi_scenario.py - -dis_scenario = 1.0 - -dis_interaction = 1.0 - -gpu_id
            # 1 > Output / movielens / movielens - dataset - 1.0 - 1.0.log 2 > & 1 &
            save_dir = '/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/douban/'
            filename = os.path.join(save_dir, f'test_dataset_{tau1}_{tau}.csv')
            # df_train.to_csv('/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/kuairand4/train_dataset.csv', index=False)
            # df_val.to_csv('/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/kuairand4/val_dataset.csv', index=False)
            #df_test.to_csv('/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/kuairand4/test_dataset_1_0.1.csv', index=False)
            df_test.to_csv(filename, index=False)
            print("end=======")
            exit()
            #print("test_dataset111", len(test_dataset))
            #     domain1_label0_train = 0
            #     domain1_label1_train = 0
            #     domain1_label0_val = 0
            #     domain1_label1_val = 0
            #     domain1_label0_test = 0
            #     domain1_label1_test = 0
            #
            #     domain2_label0_train = 0
            #     domain2_label1_train = 0
            #     domain2_label0_val = 0
            #     domain2_label1_val = 0
            #     domain2_label0_test = 0
            #     domain2_label1_test = 0
            #
            #     domain3_label0_train = 0
            #     domain3_label1_train = 0
            #     domain3_label0_val = 0
            #     domain3_label1_val = 0
            #     domain3_label0_test = 0
            #     domain3_label1_test = 0
            #     for sample in train_dataset:
            #         if sample[0]['domain_indicator'] == 0:
            #             if sample[0]['label'] == 0:
            #                 domain1_label0_train +=1
            #             else:
            #                 domain1_label1_train += 1
            #         elif sample[0]['domain_indicator'] == 1:
            #             if sample[0]['label'] == 0:
            #                 domain2_label0_train +=1
            #             else:
            #                 domain2_label1_train += 1
            #         else:
            #             if sample[0]['label'] == 0:
            #                 domain3_label0_train +=1
            #             else:
            #                 domain3_label1_train +=1
            #
            #     for sample in val_dataset:
            #         if sample[0]['domain_indicator'] == 0:
            #             if sample[0]['label'] == 0:
            #                 domain1_label0_val +=1
            #             else:
            #                 domain1_label1_val += 1
            #         elif sample[0]['domain_indicator'] == 1:
            #             if sample[0]['label'] == 0:
            #                 domain2_label0_val +=1
            #             else:
            #                 domain2_label1_val += 1
            #         else:
            #             if sample[0]['label'] == 0:
            #                 domain3_label0_val +=1
            #             else:
            #                 domain3_label1_val +=1
            #
            #     for sample in test_dataset:
            #         if sample[0]['domain_indicator'] == 0:
            #             if sample[0]['label'] == 0:
            #                 domain1_label0_test +=1
            #             else:
            #                 domain1_label1_test += 1
            #         elif sample[0]['domain_indicator'] == 1:
            #             if sample[0]['label'] == 0:
            #                 domain2_label0_test +=1
            #             else:
            #                 domain2_label1_test += 1
            #         else:
            #             if sample[0]['label'] == 0:
            #                 domain3_label0_test +=1
            #             else:
            #                 domain3_label1_test +=1
            #
            #     # domain1_train_label_ratio=[domain1_label0_train,domain1_label1_train]
            #     # domain1_val_label_ratio = [domain1_label0_val, domain1_label1_val]
            #     # domain1_test_label_ratio = [domain1_label0_test, domain1_label1_test]
            #     # #print("domain1_test_label_ratio", domain1_label0_test+domain1_label1_test)
            #     # domain2_train_label_ratio = [domain2_label0_train, domain2_label1_train]
            #     # domain2_val_label_ratio = [domain2_label0_val, domain2_label1_val]
            #     # domain2_test_label_ratio = [domain2_label0_test, domain2_label1_test]
            #     # #print("domain2_test_label_ratio", domain2_label0_test + domain2_label1_test)
            #     # domain3_train_label_ratio = [domain3_label0_train, domain3_label1_train]
            #     # domain3_val_label_ratio = [domain3_label0_val, domain3_label1_val]
            #     # domain3_test_label_ratio = [domain3_label0_test, domain3_label1_test]
            #     #print("domain3_test_label_ratio", domain3_label0_test + domain3_label1_test)
            #
            #     test_dataset_domain1_label0 = [sample for sample in test_dataset if self.condition_test_sample_domain1_label0(sample)]
            #     test_dataset_domain1_label1 = [sample for sample in test_dataset if self.condition_test_sample_domain1_label1(sample)]
            #     test_dataset_domain2_label0 = [sample for sample in test_dataset if self.condition_test_sample_domain2_label0(sample)]
            #     test_dataset_domain2_label1 = [sample for sample in test_dataset if self.condition_test_sample_domain2_label1(sample)]
            #     test_dataset_domain3_label0 = [sample for sample in test_dataset if self.condition_test_sample_domain3_label0(sample)]
            #     test_dataset_domain3_label1 = [sample for sample in test_dataset if self.condition_test_sample_domain3_label1(sample)]
            #
            #     beta = 0.8
            #
            #     if beta == 1:
            #         test_tau_domain_1 = [0, domain1_label1_test]
            #         test_tau_domain_2 = [0, domain2_label1_test]
            #         test_tau_domain_3 = [0, domain3_label1_test]
            #     else:
            #
            #         # test_tau_domain_1 = [int((1 - beta) * domain1_label0_test),
            #         #                      domain1_label0_test + domain1_label1_test - int((1 - beta) * domain1_label0_test)]
            #         if domain1_label0_test < domain1_label0_test:
            #             test_tau_domain_1 = [int((1 - beta) * domain1_label0_test),domain1_label1_test]
            #         else:
            #             test_tau_domain_1 = [domain1_label0_test, int((1 - beta) *domain1_label1_test)]
            #
            #         if domain2_label0_test < domain2_label0_test:
            #             test_tau_domain_2 = [int((1 - beta) * domain2_label0_test),domain2_label1_test]
            #         else:
            #             test_tau_domain_2 = [domain2_label0_test, int((1 - beta) *domain2_label1_test)]
            #
            #         if domain3_label0_test < domain3_label0_test:
            #             test_tau_domain_3 = [int((1 - beta) * domain3_label0_test), domain3_label1_test]
            #         else:
            #             test_tau_domain_3 = [domain3_label0_test, int((1 - beta) * domain3_label1_test)]
            #
            #     # exit()
            #     if len(test_dataset_domain1_label0) < test_tau_domain_1[0]:
            #         test_dataset_domain1_label0_sample =test_dataset_domain1_label0
            #     else:
            #         test_dataset_domain1_label0_sample = random.sample(test_dataset_domain1_label0,test_tau_domain_1[0])
            #     if len(test_dataset_domain1_label1) < test_tau_domain_1[1]:
            #         test_dataset_domain1_label1_sample = test_dataset_domain1_label1
            #     else:
            #         test_dataset_domain1_label1_sample = random.sample(test_dataset_domain1_label1, test_tau_domain_1[1])
            #     test_dataset_domain1_label_len=test_tau_domain_1[0]+test_tau_domain_1[1]
            #
            #     if len(test_dataset_domain2_label0) < test_tau_domain_2[0]:
            #         test_dataset_domain2_label0_sample =test_dataset_domain2_label0
            #     else:
            #         test_dataset_domain2_label0_sample = random.sample(test_dataset_domain2_label0, test_tau_domain_2[0])
            #     if len(test_dataset_domain2_label1) < test_tau_domain_2[1]:
            #         test_dataset_domain2_label1_sample =test_dataset_domain2_label1
            #     else:
            #         test_dataset_domain2_label1_sample = random.sample(test_dataset_domain2_label1, test_tau_domain_2[1])
            #     test_dataset_domain2_label_len =test_tau_domain_2[0]+test_tau_domain_2[1]
            #
            #     if len(test_dataset_domain3_label0) < test_tau_domain_3[0]:
            #         test_dataset_domain3_label0_sample =test_dataset_domain3_label0
            #     else:
            #         test_dataset_domain3_label0_sample = random.sample(test_dataset_domain3_label0, test_tau_domain_3[0])
            #     if len(test_dataset_domain3_label1) < test_tau_domain_3[1]:
            #         test_dataset_domain3_label1_sample =test_dataset_domain3_label1
            #     else:
            #         test_dataset_domain3_label1_sample = random.sample(test_dataset_domain3_label1, test_tau_domain_3[1])
            #     test_dataset_domain3_label_len = test_tau_domain_3[0] + test_tau_domain_3[1]
            #
            #     test_dataset_domain_len=test_dataset_domain1_label_len + test_dataset_domain2_label_len+test_dataset_domain3_label_len
            #
            #     test_dataset_domain1_sample_num = int(domain_ratio[0] * test_dataset_domain_len)
            #     test_dataset_domain2_sample_num = int(domain_ratio[1] * test_dataset_domain_len)
            #     test_dataset_domain3_sample_num = test_dataset_domain_len - test_dataset_domain1_sample_num - test_dataset_domain2_sample_num
            #
            #     # 在每个样本集中按照计算出的样本个数取样本
            #     selected_samples_a = self.sample_with_original_ratio(test_dataset_domain1_label0_sample,
            #                                                         test_dataset_domain1_label1_sample,
            #                                                         test_dataset_domain1_sample_num)
            #     selected_samples_b = self.sample_with_original_ratio(test_dataset_domain2_label0_sample,
            #                                                         test_dataset_domain2_label1_sample,
            #                                                         test_dataset_domain2_sample_num)
            #     selected_samples_c = self.sample_with_original_ratio(test_dataset_domain3_label0_sample,
            #                                                         test_dataset_domain3_label1_sample,
            #                                                         test_dataset_domain3_sample_num)
            #     test_dataset=selected_samples_a+selected_samples_b+selected_samples_c
            #
            # print("test_dataset111", len(test_dataset))
        else:
            train_dataset = self.dataset
            val_dataset = TorchDataset(x_val, y_val)
            test_dataset = TorchDataset(x_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last = drop_last_flag)
        #train_domain_distribution=self.domain_distribution(train_dataloader)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last = drop_last_flag)
        #val_domain_distribution = self.domain_distribution(val_dataloader)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last = drop_last_flag)
        #test_domain_distribution = self.domain_distribution(test_dataloader)
        # print("train_domain_distribution",train_domain_distribution)
        # print("val_domain_distribution", val_domain_distribution)
        # print("test_domain_distribution", test_domain_distribution)
        # exit()
        return train_dataloader, val_dataloader, test_dataloader

    def domain_distribution_kuairand(self,data):
        # 统计domain 分布
        id1 = 0
        id2 = 0
        id3 = 0
        id4 = 0
        for sample in data:
            domain_num = sample[0]['domain_indicator']
            for domain_id in domain_num:
                if domain_id == 0:
                    id1 += 1
                elif domain_id == 1:
                    id2 += 1
                elif domain_id == 2:
                    id3 += 1
                elif domain_id == 3:
                    id4 += 1

        result=[id1,id2,id3,id4]

        return result

    def domain_distribution(self,data):
        # 统计domain 分布
        id1 = 0
        id2 = 0
        id3 = 0
        for sample in data:
            domain_num = sample[0]['domain_indicator']
            for domain_id in domain_num:
                if domain_id == 0:
                    id1 += 1
                elif domain_id == 1:
                    id2 += 1
                elif domain_id == 2:
                    id3 += 1

        result=[id1,id2,id3]
        return result


def get_auto_embedding_dim(num_classes):
    """ Calculate the dim of embedding vector according to number of classes in the category
    emb_dim = [6 * (num_classes)^(1/4)]
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    Args:
        num_classes: number of classes in the category
    
    Returns:
        the dim of embedding vector
    """
    return np.floor(6 * np.pow(num_classes, 0.26))


def get_loss_func(task_type="classification"):
    if task_type == "classification":
        return torch.nn.BCELoss()
    elif task_type == "regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError("task_type must be classification or regression")


def get_metric_func(task_type="classification"):
    if task_type == "classification":
        return roc_auc_score
    elif task_type == "regression":
        return mean_squared_error
    else:
        raise ValueError("task_type must be classification or regression")


def generate_seq_feature(data,
                         user_col,
                         item_col,
                         time_col,
                         item_attribute_cols=[],
                         min_item=0,
                         shuffle=True,
                         max_len=50):
    """generate sequence feature and negative sample for ranking.

    Args:
        data (pd.DataFrame): the raw data.
        user_col (str): the col name of user_id
        item_col (str): the col name of item_id
        time_col (str): the col name of timestamp
        item_attribute_cols (list[str], optional): the other attribute cols of item which you want to generate sequence feature. Defaults to `[]`.
        sample_method (int, optional): the negative sample method `{
            0: "random sampling",
            1: "popularity sampling method used in word2vec",
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`.
            Defaults to 0.
        min_item (int, optional): the min item each user must have. Defaults to 0.
        shuffle (bool, optional): shulle if True
        max_len (int, optional): the max length of a user history sequence.

    Returns:
        pd.DataFrame: split train, val and test data with sequence features by time.
    """
    for feat in data:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])
        data[feat] = data[feat].apply(lambda x: x + 1)  # 0 to be used as the symbol for padding
    data = data.astype('int32')

    # generate item to attribute mapping
    n_items = data[item_col].max()
    item2attr = {}
    if len(item_attribute_cols) > 0:
        for col in item_attribute_cols:
            map = data[[item_col, col]]
            item2attr[col] = map.set_index([item_col])[col].to_dict()

    train_data, val_data, test_data = [], [], []
    data.sort_values(time_col, inplace=True)
    # Sliding window to construct negative samples
    for uid, hist in tqdm.tqdm(data.groupby(user_col), desc='generate sequence features'):
        pos_list = hist[item_col].tolist()
        len_pos_list = len(pos_list)
        if len_pos_list < min_item:  # drop this user when his pos items < min_item
            continue

        neg_list = [neg_sample(pos_list, n_items) for _ in range(len_pos_list)]
        for i in range(1, min(len_pos_list, max_len)):
            hist_item = pos_list[:i]
            hist_item = hist_item + [0] * (max_len - len(hist_item))
            pos_item = pos_list[i]
            neg_item = neg_list[i]
            pos_seq = [1, pos_item, uid, hist_item]
            neg_seq = [0, neg_item, uid, hist_item]
            if len(item_attribute_cols) > 0:
                for attr_col in item_attribute_cols:  # the history of item attribute features
                    hist_attr = hist[attr_col].tolist()[:i]
                    hist_attr = hist_attr + [0] * (max_len - len(hist_attr))
                    pos2attr = [hist_attr, item2attr[attr_col][pos_item]]
                    neg2attr = [hist_attr, item2attr[attr_col][neg_item]]
                    pos_seq += pos2attr
                    neg_seq += neg2attr
            if i == len_pos_list - 1:
                test_data.append(pos_seq)
                test_data.append(neg_seq)
            elif i == len_pos_list - 2:
                val_data.append(pos_seq)
                val_data.append(neg_seq)
            else:
                train_data.append(pos_seq)
                train_data.append(neg_seq)

    col_name = ['label', 'target_item_id', user_col, 'hist_item_id']
    if len(item_attribute_cols) > 0:
        for attr_col in item_attribute_cols:  # the history of item attribute features
            name = ['hist_'+attr_col, 'target_'+attr_col]
            col_name += name

    # shuffle
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    train = pd.DataFrame(train_data, columns=col_name)
    val = pd.DataFrame(val_data, columns=col_name)
    test = pd.DataFrame(test_data, columns=col_name)

    return train, val, test


def df_to_dict(data):
    """
    Convert the DataFrame to a dict type input that the network can accept
    Args:
        data (pd.DataFrame): datasets of type DataFrame
    Returns:
        The converted dict, which can be used directly into the input network
    """
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict


def neg_sample(click_hist, item_size):
    neg = random.randint(1, item_size)
    while neg in click_hist:
        neg = random.randint(1, item_size)
    return neg


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length.
        This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences
        reference: https://github.com/huawei-noah/benchmark/tree/main/FuxiCTR/fuxictr
    """
    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncating in ["pre", "post"], "Invalid truncating={}.".format(truncating)

    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # empty list
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr


def array_replace_with_dict(array, dic):
    """Replace values in NumPy array based on dictionary.
    Args:
        array (np.array): a numpy array
        dic (dict): a map dict

    Returns:
        np.array: array with replace
    """
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    idx = k.argsort()
    return v[idx[np.searchsorted(k, array, sorter=idx)]]


import pandas as pd
import numpy as np


def reduce_mem_usage(df):
    """Reduce memory.
    Args:
        df (pd.dataframe): a pandas dataframe
    Returns:
        df (pd.dataframe): a pandas dataframe
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df