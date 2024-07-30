import torch
import numpy as np

def get_recall(indices, targets, topk):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = torch.cumsum((targets == indices), dim=1, dtype=torch.float)
    metric_dict = {}
    for k in topk:
        metric_dict["Recall@{}".format(k)] = hits[:, k-1]
    return metric_dict


def get_NDCG(indices, targets, topk): 
    ranks = torch.zeros_like(indices, dtype=torch.float)
    ranks[:, :] = torch.arange(2, indices.shape[1] + 2)
    dcg = 1.0 / torch.log2(ranks)
    temp_zero = torch.zeros_like(dcg, dtype=torch.float).to(indices.device)
    dcg = torch.cumsum(torch.where(indices == targets.view(-1, 1).expand_as(indices), dcg, temp_zero), dim=1)
    ndcg = dcg / (1.0 / np.log2(2))
    result = {}
    for k in topk:
        result["NDCG@{}".format(k)] = ndcg[:, k-1]
    return result


def get_MRR(indices,targets,topk):
    result = {}
    for k in topk:
        mrr_sum = 0
        indices_temp = indices[:, :k]
        targets_temp = targets.view(-1, 1).expand_as(indices_temp)
        hits  = (targets_temp==indices_temp).nonzero()
        index_list = hits[:,-1]
        index_list = index_list.cpu().numpy().tolist()
        for i in index_list:
            mrr_sum += 1/(i+1)
        result["MRR_sum@{}".format(k)] = torch.tensor(mrr_sum)
        result["MRR_count@{}".format(k)] = torch.tensor(len(targets))
    return result

def evaluate(seq, scores, targets, topk=[5,20]):
    scores_top, indices = torch.topk(scores, max(topk), -1)
    result = {}
    result.update(get_recall(indices, targets, topk))
    result.update(get_NDCG(indices, targets, topk))
    result.update(get_MRR(indices,targets,topk))
    return result
