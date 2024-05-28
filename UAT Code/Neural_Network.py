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



class CustomDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        return {
            "x": torch.tensor(current_sample, dtype=torch.float),
            "y": torch.tensor(current_target, dtype=torch.float),
        }

num_points = []
quiet = False
for i in range(30):
    string = str(25*i + 25)
    num_points.append(string.zfill(6))

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU)")

for i in range(29,30):
    torch.cuda.synchronize()
    start_func = time.time()
    for j in range(1):
        if not quiet:
            print('Func_' + str(num_points[i]) + ' Started')
        import_path = 'training_data/func_' + str(num_points[i]) + '.mat'
        export_path = 'mse/func_' + str(num_points[i])
        
        # print(os.getcwd())
        mat = scipy.io.loadmat(import_path)
        data = np.array(mat['x'])[:, j]
        data = data.reshape(-1, 1)
        targets = np.array(mat['y'])[:, j]
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

        # Define the model
        width = 100

        model = nn.Sequential(
            nn.Linear(1, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )
        model = model.to(device)
        
        # loss function and optimizer
        lr_start = 0.05
        lr_final = 0.0000005
        n_epochs = 200   # number of epochs to run
        batch_size = 10  # size of each batch
        scheduled = False
        
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_start)
        if scheduled:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(lr_final/lr_start)**(1/n_epochs))
        batch_start = torch.arange(0, len(train_data), batch_size)
        
        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        times = []
        
        for epoch in range(n_epochs):
            torch.cuda.synchronize()
            start_epoch = time.time()
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=quiet) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = train_data[start:start+batch_size]
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
            if scheduled:
                scheduler.step()
            # print(scheduler.get_last_lr())
            
        
        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        if not quiet:
            print("MSE: %.6f" % best_mse)
            print("RMSE: %.6f" % np.sqrt(best_mse))
            print("TOTAL TIME: %.3f" % sum(times))

        #Save MSE as a .mat
        output_dict = {"mse": history, "epoch_times": times}
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        savemat(export_path + '/trial' + str(j) +  '.mat', output_dict)
    torch.cuda.synchronize()
    end_func = time.time()
    elapsed_time_func = end_func - start_func
    print('Func_' + str(num_points[i]) + " took " + str(elapsed_time_func) + " seconds")
    state = {'model': model, 'train_data': train_data, 'train_targets': train_targets, 'test_data': test_targets, 'test_targets': test_targets, 'state_dict': model.state_dict()}
    # torch.save(model, 'UAT Code/test_model.pt')
    # torch.save(train_targets, 'UAT Code/train_targets.pt')
    # torch.save(train_data, 'UAT Code/test_data.pt')
    # torch.save(test_targets, 'UAT Code/test_targets.pt')
    # torch.save(test_data, 'UAT Code/test_data.pt')
    # torch.save(model.state_dict(), 'UAT Code/test_state_dict.pt')
    torch.save(state, 'state1.pt')

if not quiet:
    plt.plot(history)
    plt.show()