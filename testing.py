import copy
import os
import time
import torch
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import matplotlib as plt
from scipy.io import savemat
from scipy.io import loadmat
from utils import * 
from Bezier import bezierCurve

PATH = 'complete/other/'
for i in range(1):
    # test = loadmat(PATH + 'bezier' + str(i) + str(i+1) + '.mat')
    # bezierData = test['bezier_data']
    # print('There are ' + str(np.sum(areThereNans)) + ' NaNs in bezier' + str(i) + str(i+1) + '.mat')
    # if np.sum(np.isnan(bezierData)) >= 0:
    if True:
        path1 = PATH + 'state' + str(i) + '.pt'
        path2 = PATH + 'state' + str(i+1) + '.pt'
        
        n_epochs_curve = 10000
        output_dict = bezierCurve(lr=0.001, n_epochs_curve=n_epochs_curve, batch_size=10, 
                                  quiet=False, path1=path1, path2=path2, export_path='checkpoints/')
        # print(output_dict['bezier_data'])
        if np.sum(np.isnan(output_dict['bezier_data'])) > 0:
            print('There are still NaNs in bezier' + str(i) + str(i+1) + '.mat')
        else:
            print('There are no longer NaNs in bezier' + str(i) + str(i+1) + '.mat')
            savemat(PATH + 'bezier' + str(i) + str(i+1) + '.mat', output_dict)

# print(test_targets.size())
# 
# x = torch.cat((train_data, test_data)).cpu()
# y = torch.cat((train_targets, test_targets)).cpu()
# 
# plt.title("Train Loss Along Path")
# plt.xlabel("t")
# plt.ylabel("Train Loss")
# 
# plt.scatter(train_data.cpu(), train_targets.cpu(), color="blue", label='Bezier Curve')
# plt.show()

# dict = torch.load(PATH + 'state0.pt')
# model = dict['model']
# train_data = dict['train_data']
# train_targets = dict['train_targets']
# test_data = dict['test_data']
# test_targets = dict['test_targets']
# weights = dict['state_dict']
# loss_fn = nn.MSELoss()
# 
# model.load_state_dict(weights)
# model.eval()
# y_pred = model(train_data[570:580])
# mse = loss_fn(y_pred, train_targets[570:580])
# mse = float(mse)
# print(mse)