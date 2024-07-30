import os

import torch
import tqdm
from sklearn.metrics import roc_auc_score,log_loss
from ..basic.callback import EarlyStopper
from torch.nn.utils import clip_grad_norm_
class CTRTrainerV(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model,
        learning_rate_f,
        learning_rate_v,
        learning_rate_w,
        learning_rate_c,
        scheduler_f=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer_f = torch.optim.Adam(model.baseModel.parameters(), lr=learning_rate_f,weight_decay=1e-5)
        #self.optimizer_weight_vw = torch.optim.Adam([model.V,model.W], lr=learning_rate_vw,weight_decay=1e-5)
        self.optimizer_weight_v = torch.optim.Adagrad([model.V], lr=learning_rate_v, weight_decay=1e-5) # Adagrad
        self.optimizer_weight_w = torch.optim.Adam([model.W], lr=learning_rate_w, weight_decay=1e-5)
        self.optimizer_cluster = torch.optim.Adam(model.cluster.parameters(), lr=learning_rate_c,weight_decay=1e-5)
        # if optimizer_params is None:
        #     optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        # self.optimizer_f = optimizer_fn(self.model.baseModel.parameters(), **optimizer_params)  # default optimizer

        self.scheduler = None
        if scheduler_f is not None:
            self.scheduler = scheduler_f(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()  #default loss cross_entropy
        self.evaluate_fn = roc_auc_score  #default evaluate function
        self.evaluate_fn_logloss = log_loss
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            # if i > 0:
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
            y = y.to(self.device)

            # add DRO to learn inter-level V
            loss, cluster_loss = self.model(x_dict, y)
            # print("loss",loss)
            # print("cluster_loss", cluster_loss)
            loss_1 = -loss

            self.optimizer_weight_w.zero_grad()
            self.optimizer_weight_v.zero_grad()
            #self.optimizer_weight_vw.zero_grad()
            loss_1.backward()

            self.optimizer_weight_w.step()
            self.optimizer_weight_v.step()


            self.optimizer_cluster.zero_grad()
            cluster_loss.backward()
            #clip_grad_norm_(self.model.cluster.parameters(), max_norm=10.0)
            self.optimizer_cluster.step()

            loss,_= self.model(x_dict,y)
            loss_2 = loss
            self.optimizer_f.zero_grad()
            loss_2.backward()
            #clip_grad_norm_(self.model.baseModel.parameters(), max_norm=10.0)
            self.optimizer_f.step()

            total_loss += loss_2.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            #print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                log_loss = self.evaluate_logloss(self.model, val_dataloader)
                #print('epoch:', epoch_i, 'validation: auc:', auc, 'log loss:', log_loss)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    #print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  #save best auc model

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts)

    def evaluate_logloss(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return log_loss(targets,predicts)

    def evaluate_multi_domain_logloss(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                for d in range(3):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = log_loss(targets1, predicts1) if predicts1 else None
        domain2_val = log_loss(targets2, predicts2) if predicts2 else None
        domain3_val = log_loss(targets3, predicts3) if predicts3 else None
        total_val = log_loss(targets, predicts) if predicts else None

        return domain1_val, domain2_val, domain3_val, total_val

    def evaluate_multi_domain_logloss_kuairand(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        targets4, predicts4 = list(), list()
        targets5, predicts5 = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                for d in range(5):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                y4 = y[domain_mask_list[3]].tolist()
                y_pred_4 = y_pred[domain_mask_list[3]].tolist()
                targets4.extend(y4)
                predicts4.extend(y_pred_4)

                y5 = y[domain_mask_list[4]].tolist()
                y_pred_5 = y_pred[domain_mask_list[4]].tolist()
                targets5.extend(y5)
                predicts5.extend(y_pred_5)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = log_loss(targets1, predicts1) if predicts1 else None
        domain2_val = log_loss(targets2, predicts2) if predicts2 else None
        domain3_val = log_loss(targets3, predicts3) if predicts3 else None
        domain4_val = log_loss(targets4, predicts4) if predicts4 else None
        domain5_val = log_loss(targets5, predicts5) if predicts5 else None
        total_val = log_loss(targets, predicts) if predicts else None

        return domain1_val, domain2_val, domain3_val,domain4_val, domain5_val, total_val
    def evaluate_multi_domain_logloss_kuairand4(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        targets4, predicts4 = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                for d in range(4):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                y4 = y[domain_mask_list[3]].tolist()
                y_pred_4 = y_pred[domain_mask_list[3]].tolist()
                targets4.extend(y4)
                predicts4.extend(y_pred_4)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = log_loss(targets1, predicts1) if predicts1 else None
        domain2_val = log_loss(targets2, predicts2) if predicts2 else None
        domain3_val = log_loss(targets3, predicts3) if predicts3 else None
        domain4_val = log_loss(targets4, predicts4) if predicts4 else None
        total_val = log_loss(targets, predicts) if predicts else None

        return domain1_val, domain2_val, domain3_val,domain4_val, total_val
    def evaluate_multi_domain_auc_kuairand4(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        targets4, predicts4 = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                for d in range(4):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                y4 = y[domain_mask_list[3]].tolist()
                y_pred_4 = y_pred[domain_mask_list[3]].tolist()
                targets4.extend(y4)
                predicts4.extend(y_pred_4)


                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = self.evaluate_fn(targets1, predicts1) if predicts1 else None
        domain2_val = self.evaluate_fn(targets2, predicts2) if predicts2 else None
        domain3_val = self.evaluate_fn(targets3, predicts3) if predicts3 else None
        domain4_val = self.evaluate_fn(targets4, predicts4) if predicts4 else None
        total_val   = self.evaluate_fn(targets, predicts) if predicts else None
        return domain1_val, domain2_val, domain3_val,domain4_val, total_val
    def evaluate_multi_domain_auc(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                for d in range(3):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = self.evaluate_fn(targets1, predicts1) if predicts1 else None
        domain2_val = self.evaluate_fn(targets2, predicts2) if predicts2 else None
        domain3_val = self.evaluate_fn(targets3, predicts3) if predicts3 else None
        total_val   = self.evaluate_fn(targets, predicts) if predicts else None
        return domain1_val, domain2_val, domain3_val, total_val

    def evaluate_multi_domain_auc_kuairand(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        targets4, predicts4 = list(), list()
        targets5, predicts5 = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                for d in range(5):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                y4 = y[domain_mask_list[3]].tolist()
                y_pred_4 = y_pred[domain_mask_list[3]].tolist()
                targets4.extend(y4)
                predicts4.extend(y_pred_4)

                y5 = y[domain_mask_list[4]].tolist()
                y_pred_5 = y_pred[domain_mask_list[4]].tolist()
                targets5.extend(y5)
                predicts5.extend(y_pred_5)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = self.evaluate_fn(targets1, predicts1) if predicts1 else None
        domain2_val = self.evaluate_fn(targets2, predicts2) if predicts2 else None
        domain3_val = self.evaluate_fn(targets3, predicts3) if predicts3 else None
        domain4_val = self.evaluate_fn(targets4, predicts4) if predicts4 else None
        domain5_val = self.evaluate_fn(targets5, predicts5) if predicts5 else None
        total_val   = self.evaluate_fn(targets, predicts) if predicts else None
        return domain1_val, domain2_val, domain3_val,domain4_val, domain5_val, total_val


    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model.baseModel(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

class CTRTrainer(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer

        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()  #default loss cross_entropy
        self.evaluate_fn = roc_auc_score  #default evaluate function
        self.evaluate_fn_logloss = log_loss
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
            y = y.to(self.device)

            y_pred = self.model(x_dict)
            # print("y_pred.float()", y_pred)
            loss = self.criterion(y_pred, y.float())
            #print("loss", loss)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                log_loss = self.evaluate_logloss(self.model, val_dataloader)
                print('epoch:', epoch_i, 'validation: auc:', auc, 'log loss:', log_loss)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  #save best auc model

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                # print("y.tolist()",y.tolist())
                # print("y.y_pred()", y_pred.tolist())
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts)

    def evaluate_logloss(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return log_loss(targets,predicts)

    def evaluate_multi_domain_logloss(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model(x_dict)
                for d in range(3):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = log_loss(targets1, predicts1) if predicts1 else None
        domain2_val = log_loss(targets2, predicts2) if predicts2 else None
        domain3_val = log_loss(targets3, predicts3) if predicts3 else None
        total_val = log_loss(targets, predicts) if predicts else None

        return domain1_val, domain2_val, domain3_val, total_val

    def evaluate_multi_domain_logloss_kuairand(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        targets1, predicts1 = list(), list()
        targets2, predicts2 = list(), list()
        targets3, predicts3 = list(), list()
        targets4, predicts4 = list(), list()
        targets5, predicts5 = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model(x_dict)
                for d in range(5):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                y4 = y[domain_mask_list[3]].tolist()
                y_pred_4 = y_pred[domain_mask_list[3]].tolist()
                targets4.extend(y4)
                predicts4.extend(y_pred_4)

                y5 = y[domain_mask_list[4]].tolist()
                y_pred_5 = y_pred[domain_mask_list[4]].tolist()
                targets5.extend(y5)
                predicts5.extend(y_pred_5)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = log_loss(targets1, predicts1) if predicts1 else None
        domain2_val = log_loss(targets2, predicts2) if predicts2 else None
        domain3_val = log_loss(targets3, predicts3) if predicts3 else None
        domain4_val = log_loss(targets4, predicts4) if predicts4 else None
        domain5_val = log_loss(targets5, predicts5) if predicts5 else None
        total_val = log_loss(targets, predicts) if predicts else None

        return domain1_val, domain2_val, domain3_val, domain4_val, domain5_val, total_val

    def evaluate_multi_domain_logloss_kuairand4(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        targets4, predicts4 = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model(x_dict)
                for d in range(4):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                y4 = y[domain_mask_list[3]].tolist()
                y_pred_4 = y_pred[domain_mask_list[3]].tolist()
                targets4.extend(y4)
                predicts4.extend(y_pred_4)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = log_loss(targets1, predicts1) if predicts1 else None
        domain2_val = log_loss(targets2, predicts2) if predicts2 else None
        domain3_val = log_loss(targets3, predicts3) if predicts3 else None
        domain4_val = log_loss(targets4, predicts4) if predicts4 else None
        total_val = log_loss(targets, predicts) if predicts else None

        return domain1_val, domain2_val, domain3_val,domain4_val, total_val
    def evaluate_multi_domain_auc_kuairand4(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        targets4, predicts4 = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model(x_dict)
                for d in range(4):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                y4 = y[domain_mask_list[3]].tolist()
                y_pred_4 = y_pred[domain_mask_list[3]].tolist()
                targets4.extend(y4)
                predicts4.extend(y_pred_4)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = self.evaluate_fn(targets1, predicts1) if predicts1 else None
        domain2_val = self.evaluate_fn(targets2, predicts2) if predicts2 else None
        domain3_val = self.evaluate_fn(targets3, predicts3) if predicts3 else None
        domain4_val = self.evaluate_fn(targets4, predicts4) if predicts4 else None
        total_val   = self.evaluate_fn(targets, predicts) if predicts else None
        return domain1_val, domain2_val, domain3_val,domain4_val, total_val
    def evaluate_multi_domain_auc(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model(x_dict)
                for d in range(3):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = self.evaluate_fn(targets1, predicts1) if predicts1 else None
        domain2_val = self.evaluate_fn(targets2, predicts2) if predicts2 else None
        domain3_val = self.evaluate_fn(targets3, predicts3) if predicts3 else None
        total_val   = self.evaluate_fn(targets, predicts) if predicts else None

        return domain1_val, domain2_val, domain3_val, total_val

    def evaluate_multi_domain_auc_kuairand(self, model, data_loader):
        model.eval()
        targets, predicts   = list() ,list()
        targets1, predicts1 = list() ,list()
        targets2, predicts2 = list() ,list()
        targets3, predicts3 = list() ,list()
        targets4, predicts4 = list(), list()
        targets5, predicts5 = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model(x_dict)
                for d in range(5):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                y1 = y[domain_mask_list[0]].tolist()
                y_pred_1 = y_pred[domain_mask_list[0]].tolist()
                targets1.extend(y1)
                predicts1.extend(y_pred_1)

                y2 = y[domain_mask_list[1]].tolist()
                y_pred_2 = y_pred[domain_mask_list[1]].tolist()
                targets2.extend(y2)
                predicts2.extend(y_pred_2)

                y3 = y[domain_mask_list[2]].tolist()
                y_pred_3 = y_pred[domain_mask_list[2]].tolist()
                targets3.extend(y3)
                predicts3.extend(y_pred_3)

                y4 = y[domain_mask_list[3]].tolist()
                y_pred_4 = y_pred[domain_mask_list[3]].tolist()
                targets4.extend(y4)
                predicts4.extend(y_pred_4)

                y5 = y[domain_mask_list[4]].tolist()
                y_pred_5 = y_pred[domain_mask_list[4]].tolist()
                targets5.extend(y5)
                predicts5.extend(y_pred_5)

                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        domain1_val = self.evaluate_fn(targets1, predicts1) if predicts1 else None
        domain2_val = self.evaluate_fn(targets2, predicts2) if predicts2 else None
        domain3_val = self.evaluate_fn(targets3, predicts3) if predicts3 else None
        domain4_val = self.evaluate_fn(targets4, predicts4) if predicts4 else None
        domain5_val = self.evaluate_fn(targets5, predicts5) if predicts5 else None
        total_val   = self.evaluate_fn(targets, predicts) if predicts else None
        return domain1_val, domain2_val, domain3_val,domain4_val, domain5_val, total_val
    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

class CTRTrainerMultiDomain(object):
    """A general trainer for single task multi domain learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  #default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()  #default loss cross_entropy
        self.evaluate_fn = roc_auc_score  #default evaluate function
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU

            y = y.to(self.device)
            y_pred = self.model(x_dict)

            loss = self.criterion(y_pred.reshape(-1), y.reshape(-1).float())
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)

            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                print('epoch:', epoch_i, 'validation: auc:', auc)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  #save best auc model

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()

        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)

                y_pred = model(x_dict)

                domain1_pre = []
                domain1_target = []
                domain2_pre = []
                domain2_target = []
                domain3_pre = []
                domain3_target = []

                domain1_target.extend(y[:,0].reshape(-1).tolist())
                domain2_target.extend(y[:, 1].reshape(-1).tolist())
                domain3_target.extend(y[:, 2].reshape(-1).tolist())

                domain1_pre.extend(y_pred[:,0].reshape(-1).tolist())
                domain2_pre.extend(y_pred[:,1].reshape(-1).tolist())
                domain3_pre.extend(y_pred[:,2].reshape(-1).tolist())

                targets.extend(y.reshape(-1).tolist())
                predicts.extend(y_pred.reshape(-1).tolist())
            print("domain1 auc:{}".format(self.evaluate_fn(domain1_target, domain1_pre)))
            print("domain2 auc:{}".format(self.evaluate_fn(domain2_target, domain2_pre)))
            print("domain3 auc:{}".format(self.evaluate_fn(domain3_target, domain3_pre)))
        return self.evaluate_fn(targets, predicts)

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device).re
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts