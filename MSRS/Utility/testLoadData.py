import torch
from torch.utils.data import Dataset
import random
import sys, os
os.chdir(sys.path[0])

def load_data_general(root,n_items,noisy_item_num=0,noisy_pre=0):
    path_train_data = root + 'train_noisy_'+str(noisy_pre)+'.txt'
    path_test_data = root + 'test.txt'
    # check
    if os.path.exists(path_train_data) == False:
        create_train_test_data(root=root,n_items=n_items,noisy_item_num=noisy_item_num,precentage=noisy_pre)
    train_set = []
    valid_set = []
    train_inputList = []
    train_targetList = []
    valid_inputList = []
    valid_targetList = []
    with open(path_train_data, 'r') as f1:
        for line in f1:
            temp = list(map(int,line.strip("\n").split(",")))
            train_x = temp[:-2]
            train_y = temp[-2]
            valid_x = temp[:-1]
            valid_y = temp[-1]

            train_inputList.append(train_x)
            train_targetList.append(train_y)

            valid_inputList.append(valid_x)
            valid_targetList.append(valid_y)

    train_set.append(train_inputList)
    train_set.append(train_targetList)
    train = train_set
    valid_set.append(valid_inputList)
    valid_set.append(valid_targetList)
    valid = valid_set

    test_set = []
    xlist = []
    ylist = []
    with open(path_test_data, 'r') as f1:
        for line in f1:
            temp = list(map(int,line.strip("\n").split(",")))
            inputs = temp[:-1]
            target = temp[-1]
            xlist.append(inputs)
            ylist.append(target)
    test_set.append(xlist)
    test_set.append(ylist)
    test = test_set
    return train, valid, test

def create_train_test_data(root,n_items,noisy_item_num,precentage):
    maxlen = 20
    minlen = 2
    noisy_item_num = noisy_item_num
    filename = 'Source.txt'
    path_src_data = root + filename
    train_set = []
    inputList = []
    targetList = []
    testList = []
    with open(path_src_data, 'r') as f1:
        for line in f1:
            temp = line.strip("\n").split(",")
            if len(temp) >= maxlen + 3:
                inputs = temp[:maxlen]
                target = temp[maxlen:maxlen + 3]
            else:
                inputs = temp[:-3]
                target = temp[-3:]
            test_temp = inputs + target
            testList.append(test_temp)
            x_temp = []
            for i in inputs:
                x_temp.append(int(i))
            y_temp = []
            for j in target:
                y_temp.append(int(j))
            inputList.append(x_temp)
            targetList.append(y_temp)
    train_set.append(inputList)
    train_set.append(targetList)
    new_train_set_x = []
    new_train_set_y = []
    for x, y in zip(train_set[0], train_set[1]):
        new_train_set_x.append(x)
        new_train_set_y.append(y)

    for x in new_train_set_x:
        if len(x) <= minlen:
            continue
        n_changes = round(len(x) * precentage / 100)
        changed_index = []
        for j in range(n_changes):
            change_index = random.randint(0, len(x) - 1)
            while True:
                if change_index not in changed_index:
                    changed_index.append(change_index)
                    break
                else:
                    change_index = random.randint(0, len(x) - 1)
            change_item = random.randint(n_items,n_items +noisy_item_num-1)
            while True:
                if change_item not in x:
                    break
                else:
                    change_item = random.randint(n_items, n_items + noisy_item_num-1)
            x.insert(change_index, change_item)
            del x[change_index + 1]
    train_filename = 'train' + "_noisy_" + str(precentage) + '.txt'
    train_filePath = root+train_filename
    if os.path.exists(train_filePath) == False:
        with open(train_filePath, 'w') as fout:
            for i in range(len(new_train_set_y)):
                seq_out = ""
                for j in range(len(new_train_set_x[i])):
                    temp = new_train_set_x[i][j]
                    seq_out = seq_out + str(temp) + ','
                train_target = new_train_set_y[i][0]
                valid_target = new_train_set_y[i][1]
                seq_out = seq_out + str(train_target) + "," + str(valid_target) + "\n"
                fout.write(seq_out)
    test_filename = 'test.txt'
    test_filePath = root+test_filename
    if os.path.exists(test_filePath) == False:
        with open(root + test_filename, 'w') as fout:
            for i in range(len(testList)):
                seq_out = ""
                for j in range(len(testList[i])):
                    temp = testList[i][j]
                    if j != len(testList[i]) - 1:
                        seq_out = seq_out + str(temp) + ','
                    else:
                        seq_out = seq_out + str(temp) + '\n'
                fout.write(seq_out)

def collate_fn(data):
    seq_max_len = 22
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    padded_sesss = torch.zeros(len(data), seq_max_len).long()
    for i, (sess, label) in enumerate(data):
        padded_sesss[i, :lens[i]] = torch.LongTensor(sess)
        labels.append(label)
    padded_sesss = padded_sesss.transpose(0, 1)
    return padded_sesss, torch.tensor(labels).long(), lens

class RecSysDataset(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets."""
    def __init__(self, data):
        self.data = data
        print('-' * 50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-' * 50)

    def __getitem__(self, index):
        session_items = self.data[0][index]  # list 历史交互商品id如[23,12,33]
        target_item = self.data[1][index]  # int  目标商品的id
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])