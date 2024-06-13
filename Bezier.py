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
from utils import *
from torch.nn.functional import normalize

def bezierCurve(lr, n_epochs_curve, batch_size, quiet, path1, path2, export_path):
    # print(os.getcwd())
    if not quiet:
        print("Loading Path 1: " + path1)
    dict = torch.load(path1)
    model = dict['model']
    train_data = dict['train_data']
    # train_data = normalize(train_data, dim=0)
    train_targets = dict['train_targets']
    # train_targets = normalize(train_targets, dim=0)
    test_data = dict['test_data']
    # test_data = normalize(test_data, dim=0)
    test_targets = dict['test_targets']
    # test_targets = normalize(test_targets, dim=0)
    weights_1 = copy.deepcopy(dict['state_dict'])
    
    if not quiet:
        print("Loading Path 2: " + path2)
    dict_2 = torch.load(path2)
    weights_2 = copy.deepcopy(dict_2['state_dict'])

    theta = copy.deepcopy(weights_2)
    phi = copy.deepcopy(weights_2)

    # loss function and optimizer
    # lr = 0.05
    # n_epochs = 200   # number of epochs to run
    # batch_size = batch_size  # size of each batch
    # quiet = False
    n_epochs = n_epochs_curve
    loss_fn = torch.nn.MSELoss()  # mean square error

    batch_start = torch.arange(0, len(train_data), batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # print(weights_1['0.weight'][0:3] - weights_2['0.weight'][0:3])
        torch.cuda.synchronize()
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=quiet) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                t = np.random.random_sample()
                c1 = (1-t)**2
                c = 2 * (1-t) * t
                c2 = t**2
                for key in phi:
                    phi[key] = c1 * weights_1[key] + c * theta[key] + c2 * weights_2[key]
                model.load_state_dict(phi)
                # take a batch
                X_batch = train_data[start:start+batch_size]
                y_batch = train_targets[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # print(grad_dict)
                # for p in model.parameters():
                #     p.grad *= c  
                # update weights
                optimizer.step()
                for k in theta:
                    layer_num = int(k[0])
                    layer_type = k[2:]
                    layer = getattr(model[layer_num], layer_type)
                    grads = layer.grad
                    theta[k] = theta[k] - grads * c * lr
                # print progress
                bar.set_postfix(mse=float(loss))
    # state = {'model': model, 'train_data': train_data, 'train_targets': train_targets, 'test_data': test_targets, 'test_targets': test_targets, 'state_dict': model.state_dict(), 'theta':theta}
    # torch.save(state, 'bezier.pt')
    
    mse_array_bezier = []
    mse_array_bezier_test = []
    t_array = np.arange(0, 1.001, 0.001)
    for t in t_array:
        c1 = (1-t) ** 2
        c = 2 * (1-t) * t
        c2 = t ** 2
        for key in phi:
            phi[key] = c1 * weights_1[key] + c * theta[key] + c2 * weights_2[key]
        model.load_state_dict(phi)
        model.eval()
        
        y_pred = model(train_data)
        mse = loss_fn(y_pred, train_targets)
        mse = float(mse)
        mse_array_bezier.append(mse)
        
        y_pred_test = model(test_data)
        mse_test = loss_fn(y_pred_test, test_targets)
        mse_test = float(mse_test)
        mse_array_bezier_test.append(mse_test)

    mse_array_straight = []
    mse_array_straight_test = []
    for t in t_array:
        c1 = (1-t)
        c2 = t
        for key in phi:
            phi[key] = c1 * weights_1[key] + c2 * weights_2[key]
        model.load_state_dict(phi)
        model.eval()
        
        y_pred = model(train_data)
        mse = loss_fn(y_pred, train_targets)
        mse = float(mse)
        mse_array_straight.append(mse)
        
        y_pred_test = model(test_data)
        mse_test = loss_fn(y_pred_test, test_targets)
        mse_test = float(mse_test)
        mse_array_straight_test.append(mse_test)

    # print(weights_1['0.weight'] - weights_2['0.weight'])

    plt.title("Train Loss Along Path")
    plt.xlabel("t")
    plt.ylabel("Train Loss")

    max_arg_bezier = np.argmax(mse_array_bezier)
    max_arg_straight = np.argmax(mse_array_straight)
    if not quiet:
        print('Maximum Bezier Curve Loss: '+ str(mse_array_bezier[max_arg_bezier]) + ' at t = ' + str(t_array[max_arg_bezier]))
        print('Maximum Straight Line Loss: '+ str(mse_array_straight[max_arg_straight]) + ' at t = ' + str(t_array[max_arg_straight]))

    plt.plot(t_array, mse_array_bezier, color="blue", label='Bezier Curve')
    plt.plot(t_array, mse_array_straight, color="red", label='Straight Line')
    plt.legend()

    # plt.show() 
    output_dict = {'t': t_array, 'straight_data': mse_array_straight, 'bezier_data': mse_array_bezier,
                    'bezier_data_test': mse_array_bezier_test, 'straight_data_test': mse_array_straight_test}
    savemat(export_path + 'bezier' + path1[-4] + path2[-4] + '.mat', output_dict)
    return output_dict