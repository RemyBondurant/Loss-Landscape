import argparse
import copy
import time
import os
import tqdm
import torch

import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import savemat
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.functional import normalize

from Bezier import bezierCurve

parser = argparse.ArgumentParser(description='Mode Connectivity')
parser.add_argument('--dir', type=str, default='checkpoints/', metavar='DIR',
                    help='training directory (default: checkpoints/)')
parser.add_argument('--data_path', type=str, default='training_data/', metavar='DATASET',
                    help='location of dataset (default: training_data/)')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size (default: 10)')
parser.add_argument('--curve', type=bool, default=True, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--epochs_points', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--epochs_curve', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--quiet', type=bool, default=False, metavar='QUIET',
                    help='should the program be quiet (default:False)')
parser.add_argument('--checkpoint', type=str, default=None, metavar='CHECK',
                    help='resume from checkpoint (default:None)')

parser.set_defaults(init_linear=True)

args = parser.parse_args()
print(os.getcwd())
quiet = args.quiet
lr = args.lr
n_epochs = args.epochs_points   # number of epochs to run
n_epochs_curve = args.epochs_curve # numer of epochs to find curve
batch_size = args.batch_size  # size of each batch

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU)")

for i in range(2):
    import_path = args.data_path
    export_path = args.dir
    # Define the model
    model = nn.Sequential(
        # nn.BatchNorm1d(1),
        nn.Linear(1, 5000),
        nn.ReLU(),
        nn.Linear(5000, 5000),
        nn.ReLU(),
        nn.Linear(5000, 1),
        )
    model = model.to(device)
    
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if args.checkpoint is not None:
        if not quiet:
            print("Loading Previous Checkpoint: " + args.checkpoint)
        dict = torch.load(args.checkpoint)
        model = dict['model']
        train_data = dict['train_data']
        train_targets = dict['train_targets']
        test_data = dict['test_data']
        test_targets = dict['test_targets']
        state_dict = dict['state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(dict['optimizer_state_dict'])
    else:
        if not quiet:
            print('Started ' + import_path + ' Point ' + str(i))
        mat = scipy.io.loadmat(import_path)
        data = np.array(mat['x'])[:, 0]
        data = data.reshape(-1, 1)
        targets = np.array(mat['y'])[:, 0]
        targets = targets.reshape(-1, 1)

        train_data, test_data, train_targets, test_targets = train_test_split(data, targets, train_size=0.8)
        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        train_targets = torch.tensor(train_targets, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)

        train_data = train_data.to(device)
        test_data = test_data.to(device)
        train_targets = train_targets.to(device)
        test_targets = test_targets.to(device)

        
    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    times = []
    batch_start = torch.arange(0, len(train_data), batch_size)
    torch.cuda.synchronize()
    start_func = time.time()
    
    for epoch in range(n_epochs):
        torch.cuda.synchronize()
        start_epoch = time.time()
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=quiet) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = train_data[start:start+batch_size]
                X_batch = normalize(X_batch)
                y_batch = train_targets[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        scheduler.step()
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(test_data)
        mse = loss_fn(y_pred, test_targets)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed_time = end_epoch - start_epoch
        times.append(elapsed_time)
        
    
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    if not quiet:
        print("MSE: %.6f" % best_mse)
        print("RMSE: %.6f" % np.sqrt(best_mse))
        print("TOTAL TIME: %.3f" % sum(times))

    #Save MSE as a .mat
    # output_dict = {"mse": history, "epoch_times": times}
    # if not os.path.exists(export_path):
    #     os.makedirs(export_path)
    # savemat(export_path + '/trial' + str(0) +  '.mat', output_dict)
    torch.cuda.synchronize()
    end_func = time.time()
    elapsed_time_func = end_func - start_func
    print(import_path + " took " + str(elapsed_time_func) + " seconds")
    state = {'model': model, 'train_data': train_data, 'train_targets': train_targets, 'test_data': test_data, 'test_targets': test_targets,
             'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}
    print('Exporting to: ' + export_path + 'state' + str(i) + '.pt')
    torch.save(state, export_path + 'state' + str(i) + '.pt')

    # if not quiet:
    #     plt.plot(history)
    #     plt.show()
    
if args.curve is True:
    path1 = None
    path2 = None
    for filename in os.listdir(export_path):
        f = os.path.join(export_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and f[-2:] == 'pt':
            path2 = f
            if path1 is not None:
                bezierCurve(lr, n_epochs_curve, batch_size, quiet, path1, path2, export_path)
            path1 = f