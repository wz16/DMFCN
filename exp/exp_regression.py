from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Exp_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Regression, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = 1
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        print("model parameters count: {}".format(params_count(model)))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = 0
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            count = 0
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.cpu())
                total_loss += batch_x.shape[0]*loss
                count += batch_x.shape[0]

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = total_loss/count

        # preds = torch.cat(preds, 0)
        # trues = torch.cat(trues, 0)
        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        # trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = 0
            count = 0

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label)
                train_loss += loss*batch_x.shape[0]
                count += batch_x.shape[0]

                # print("loss:{}".format(loss.item()))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # for param in self.model.parameters():
                #     if param.grad != None and torch.isnan(param.grad).any():
                #         print("Parameter {} has a NaN gradient.".format(param.name))
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = train_loss/count
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Test Loss: {4:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # if (epoch + 1) % 5 == 0:
            #     adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    def test_debug(self, setting, test=0, checkpoint=''):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            if checkpoint:
                self.model.load_state_dict(torch.load(checkpoint))            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        total_loss = 0
        criterion = self._select_criterion()
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            count = 0
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)
                # loss = criterion(outputs.cpu(), label.squeeze().cpu())

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.cpu())

                total_loss += loss*batch_x.shape[0]
                count += batch_x.shape[0]
                
                

            total_loss = total_loss/count

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/regression/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('mse_err:{}'.format(total_loss))
        f = open(folder_path+self.args.log_name, 'a')
        f.write("task:")
        f.write(setting)
        f.write(';mse_err:{}'.format(total_loss))
        f.write('\n')
        # f.write('\n')
        f.close()
        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        total_loss = 0
        criterion = self._select_criterion()
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            count = 0
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)
                # loss = criterion(outputs.cpu(), label.squeeze().cpu())

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.cpu())

                total_loss += loss*batch_x.shape[0]
                count += batch_x.shape[0]
                
                

            total_loss = total_loss/count

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/regression/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('mse_err:{}'.format(total_loss))
        f = open(folder_path+self.args.log_name, 'a')
        f.write("task:")
        f.write(setting)
        f.write(';mse_err:{}'.format(total_loss))
        f.write('\n')
        # f.write('\n')
        f.close()
        return
