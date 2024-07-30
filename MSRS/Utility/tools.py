import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
import os
#import utility.metrics as metrics
import numpy as np


def enlargeSeq(seq,target,lens):
    seq_list = seq.cpu().T.numpy().tolist()
    target_list = target.cpu().numpy().tolist()
    lens_list = lens
    max_len = 22
    new_seq_1 = []
    new_seq_2 = []
    new_target = []
    new_lens_1 = []
    new_lens_2 = []
    for i in range(len(seq_list)):
        seqlen = lens[i]
        for j in range(seqlen):
            new_seq_1.append(seq_list[i])
            new_lens_1.append(lens_list[i])
            padding_seq = seq_list[i][:seqlen-j]
            for k in range(max_len-seqlen+j):
                padding_seq.append(0)
            new_seq_2.append(padding_seq)
            new_lens_2.append(seqlen-j)
            if j == 0:
                new_target.append(target_list[j])
            else:
                new_target.append(seq_list[i][seqlen-j])
    new_seq_1 = torch.Tensor(new_seq_1).long()
    new_seq_2 = torch.Tensor(new_seq_2).long()
    new_target = torch.Tensor(new_target).long()
    return new_seq_1.T,new_seq_2.T,new_target,new_lens_1,new_lens_2


def validate(valid_loader, model,device,topk=[20]):
    model.eval()
    metrics_result = collections.defaultdict(list)
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            scores = model.baseModel(seq, lens)
            test_loss = model.loss(scores, target)
            sum_loss = torch.sum(test_loss)
            loss_value = sum_loss.item()
            metrics_result["loss"].append(loss_value)
            logits = F.softmax(scores, dim=1)
            result_dict = metrics.evaluate(seq, logits, target, topk)
            for k in result_dict.keys():
                metrics_result[k].append(torch.sum(result_dict[k]).item())
    temp_dict = {}
    for k, v in metrics_result.items():
        if k == 'MRR_sum@20':
            temp_dict['MRR@20'] = round(np.sum(v) / np.sum(metrics_result['MRR_count@20']), 4)
        elif k == 'MRR_count@20':
            continue
        else:
            metrics_result[k] = round(np.sum(v) / len(valid_loader.dataset), 4)
    metrics_result.pop('MRR_sum@20')
    metrics_result.pop('MRR_count@20')
    metrics_result['MRR@20'] = temp_dict.pop('MRR@20')
    return metrics_result

def saveModel(epoch,model,path='best_checkpoint.pth.tar',optimizer=None):
    if optimizer:
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer':optimizer.state_dict()
        }
    else:
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }
    torch.save(ckpt_dict, path)

def getSeqDict(train_loader):
    seq_dict = {}
    seq_index = 0
    for i, inter in tqdm(enumerate(train_loader), total=len(train_loader)):
        user_id = inter[0]['user_id']
        item_id = inter[0]['movie_id']
        # user_list = user_id.T.numpy().tolist()
        # item_list = item_id.T.numpy().tolist()
        user_list = user_id.permute(*torch.arange(user_id.ndim - 1, -1, -1)).numpy().tolist()
        item_list = item_id.permute(*torch.arange(item_id.ndim - 1, -1, -1)).numpy().tolist()
        for i in range(len(user_id)):
            user_item =[]
            user = user_list[i]
            item = item_list[i]
            user_item.append(user)
            user_item.append(item)
            str_interaction="_".join(map(str,filter(None, user_item)))
            if str_interaction not in seq_dict.keys():
                seq_dict.update({str_interaction: seq_index})
                seq_index += 1
    return seq_index, seq_dict
        # seq_list = seq.T.numpy().tolist()
        # for sequence in seq_list:
        #     str_seq = "_".join(map(str, filter(None, sequence)))
        #     if str_seq not in seq_dict.keys():
        #         seq_dict.update({str_seq: seq_index})
        #         seq_index += 1
    # return seq_index, seq_dict
    # for i, (seq, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
    #     seq_list = seq.T.numpy().tolist()
    #     for sequence in seq_list:
    #         str_seq = "_".join(map(str, filter(None, sequence)))
    #         if str_seq not in seq_dict.keys():
    #             seq_dict.update({str_seq: seq_index})
    #             seq_index += 1
    # return seq_index, seq_dict

def getSeqDict_kuairand(train_loader):
    seq_dict = {}
    seq_index = 0
    for i, inter in tqdm(enumerate(train_loader), total=len(train_loader)):
        user_id = inter[0]['user_id']
        item_id = inter[0]['item_id']
        # user_list = user_id.T.numpy().tolist()
        # item_list = item_id.T.numpy().tolist()
        user_list = user_id.permute(*torch.arange(user_id.ndim - 1, -1, -1)).numpy().tolist()
        item_list = item_id.permute(*torch.arange(item_id.ndim - 1, -1, -1)).numpy().tolist()
        for i in range(len(user_id)):
            user_item =[]
            user = user_list[i]
            item = item_list[i]
            user_item.append(user)
            user_item.append(item)
            str_interaction="_".join(map(str,filter(None, user_item)))
            if str_interaction not in seq_dict.keys():
                seq_dict.update({str_interaction: seq_index})
                seq_index += 1
    return seq_index, seq_dict

def getSeqDict_kuairand4(train_loader):
    seq_dict = {}
    seq_index = 0
    for i, inter in tqdm(enumerate(train_loader), total=len(train_loader)):
        user_id = inter[0]['user_id']
        item_id = inter[0]['item_id']
        # user_list = user_id.T.numpy().tolist()
        # item_list = item_id.T.numpy().tolist()
        user_list = user_id.permute(*torch.arange(user_id.ndim - 1, -1, -1)).numpy().tolist()
        item_list = item_id.permute(*torch.arange(item_id.ndim - 1, -1, -1)).numpy().tolist()
        for i in range(len(user_id)):
            user_item =[]
            user = user_list[i]
            item = item_list[i]
            user_item.append(user)
            user_item.append(item)
            str_interaction="_".join(map(str,filter(None, user_item)))
            if str_interaction not in seq_dict.keys():
                seq_dict.update({str_interaction: seq_index})
                seq_index += 1
    return seq_index, seq_dict
def getSeqDict_clustered(train_loader,n_items,k=20):
    seq_emb_dict = {}
    seq_clustered_dict = {}
    embedding_dim = 50
    F_emb = nn.Embedding(n_items, embedding_dim, padding_idx=0)
    F_GRU = nn.GRU(embedding_dim, 2)
    ClusterInput = []
    for i, (seq, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq_list = seq.T.numpy().tolist()
        for sequence in seq_list:
            str_seq = seq2str(sequence)
            seq_tensor = torch.tensor(sequence)
            seq_emb = F_emb(seq_tensor)
            seq_emb = seq_emb.unsqueeze(0)
            seq_emb = seq_emb.transpose(0, 1)
            _, seq_hidden = F_GRU(seq_emb)
            seq_cluster_emb = seq_hidden[-1][0].tolist()
            ClusterInput.append(seq_cluster_emb)
            if str_seq not in seq_emb_dict.keys():
                seq_emb_dict.update({str_seq: seq_cluster_emb})
    X = np.array(ClusterInput)
    km = KMeans(n_clusters=k,
                random_state=1,
                max_iter=200).fit(X)
    for key in seq_emb_dict.keys():
        seq_cluster_emb = seq_emb_dict[key]
        seq_class = km.predict([seq_cluster_emb])
        if key not in seq_clustered_dict.keys():
            seq_clustered_dict.update({key: seq_class[0]})
        else:
            continue
    saveName = '\seq_dict.txt'
    with open(os.getcwd()+saveName,'w') as f:
        for key in seq_clustered_dict.keys():
            value = seq_clustered_dict[key]
            outstr = str(key)+" : "+str(value)+"\n"
            f.write(outstr)
    return seq_clustered_dict


def readSeqDict(dictPath):
    seqDict ={}
    with open(dictPath,'r') as fin:
        for line in fin:
            temp  = line.strip("\n").split("\t")
            str_seq =  temp[0]
            seq_class = temp[1].split(":")[1]
            seqDict.update({str_seq:seq_class})
    return seqDict

def seq2str(sequence):
    tempList = []
    for x in sequence:
        if x != 0:
            tempList.append(str(x))
    str_seq = "_".join(tempList)
    return str_seq

def getItemDict(train_loader):
    item_dict={}
    s_index = 0
    s_init_data = []
    for i, (seq, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq_list = seq.T.numpy().tolist()
        for sequence in seq_list:
            str_seq = seq2str(sequence)
            seq_lengths = len(str_seq.split("_"))
            value = 1/seq_lengths
            s_init_data += [value]*seq_lengths
            if str_seq not in item_dict.keys():
                temp = (s_index, s_index + seq_lengths - 1)
                item_dict.update({str_seq: temp})
                s_index += seq_lengths
    return s_index,item_dict,s_init_data
