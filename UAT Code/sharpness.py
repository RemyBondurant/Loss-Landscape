import copy
import torch
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

std = 0.01
p = 100
eps = 10^-3
quiet = False

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU)")

model = torch.load('UAT Code/test_model.pt')
test_targets = torch.load('UAT Code/test_targets.pt')
test_data = torch.load('UAT Code/test_data.pt')
weights = torch.load('UAT Code/test_state_dict.pt', weights_only=True)
adjusted_weights = copy.deepcopy(weights)
num_dims = 0
flattened_weights = torch.empty(0)
flattened_weights = flattened_weights.to(device)
loss_fn = torch.nn.MSELoss()  # mean square error

for key in weights:
    num_dims += torch.numel(weights[key])
    flatened_tensor = torch.flatten(weights[key])
    flattened_weights = torch.cat((flattened_weights, flatened_tensor))

model.load_state_dict(weights)
model.eval()
y_pred = model(test_data)
mse = loss_fn(y_pred, test_targets)
mse = float(mse)
A = torch.randn(num_dims, p) * std
A = A.to(device)
A_pinv = torch.pinverse(A)
C_e = eps*torch.abs(A_pinv @ flattened_weights + 1) 
# print(C_e)
highest_mse = -np.inf
for i in range(C_e.size(dim=0)):
    new_weights_flattened = flattened_weights + A @ C_e
    start_index = 0
    for key in weights:
            tensors = weights[key]
            desired_shape = tensors.size()
            num_elems = tensors.numel()
            end_index = start_index + num_elems
            new_weight = new_weights_flattened[start_index:end_index]
            start_index = end_index
            new_weight = new_weight.reshape(desired_shape)
            # print(new_weight)
            adjusted_weights[key] = new_weight
    model.load_state_dict(adjusted_weights)
    model.eval()
    y_pred = model(test_data)
    mse_new = loss_fn(y_pred, test_targets)
    mse_new = float(mse_new)
    # print(mse_new)
    if mse_new > highest_mse:
        highest_mse = mse_new
sharpness = (highest_mse - mse)/(1 + mse) * 100
# print('------------------------')
if not quiet:
    print(highest_mse)
    print(mse)
print(sharpness)
    