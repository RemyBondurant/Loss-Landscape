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
from utils import * 

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU)")

# print(os.getcwd())
dict = torch.load('state1.pt')
model = dict['model']
train_data = dict['train_data']
train_targets = dict['train_targets']
test_data = dict['test_data']
test_targets = dict['test_targets']
weights_1 = copy.deepcopy(dict['state_dict'])

dict_2 = torch.load('state2.pt')
weights_2 = copy.deepcopy(dict_2['state_dict'])

theta = copy.deepcopy(weights_2)
phi = copy.deepcopy(weights_2)

# loss function and optimizer
lr = 0.05
n_epochs = 200   # number of epochs to run
batch_size = 10  # size of each batch
quiet = False
loss_fn = torch.nn.MSELoss()  # mean square error

batch_start = torch.arange(0, len(train_data), batch_size)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
times = []

start_func = time.time()
for epoch in range(n_epochs):
    # print(weights_1['0.weight'][0:3] - weights_2['0.weight'][0:3])
    torch.cuda.synchronize()
    start_epoch = time.time()
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
            grad_dict = {k:v.grad for k,v in model.named_parameters()}
            for k in theta:
                theta[k] = theta[k] - grad_dict[k] * c
            # print progress
            bar.set_postfix(mse=float(loss))


mse_array_bezier = []
t_array = np.arange(0, 1.001, 0.001)
for t in t_array:
    c1 = (1-t)**2
    c = 2 * (1-t) * t
    c2 = t**2
    for key in phi:
        phi[key] = c1 * weights_1[key] + c * theta[key] + c2 * weights_2[key]
    model.load_state_dict(phi)
    model.eval()
    y_pred = model(test_data)
    mse = loss_fn(y_pred, test_targets)
    mse = float(mse)
    mse_array_bezier.append(mse)

# print(weights_1['0.weight'] - weights_2['0.weight'])

mse_array_straight = []
for t in t_array:
    c1 = (1-t)
    c2 = t
    for key in phi:
        phi[key] = c1 * weights_1[key] + c2 * weights_2[key]
    model.load_state_dict(phi)
    model.eval()
    y_pred = model(test_data)
    mse = loss_fn(y_pred, test_targets)
    mse = float(mse)
    mse_array_straight.append(mse)

# print(weights_1['0.weight'] - weights_2['0.weight'])

plt.title("Loss Along Path")
plt.xlabel("t")
plt.ylabel("Loss")

print('T: ' + str(t_array[0]) + ' --> MSE: ' + str(mse_array_bezier[0]))
print('T: ' + str(t_array[-1]) + ' --> MSE: ' + str(mse_array_bezier[-1]))

plt.plot(t_array, mse_array_bezier, color="blue", label='bezier')
plt.plot(t_array, mse_array_straight, color="red", label='straight')
plt.legend()

plt.show() 