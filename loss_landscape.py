import copy
import os
import torch
import numpy as np
import scipy.io
import pandas as pd
import numpy as np
from utils import * 
    
num_lengths = 100
num_directions = 10
std = 1
p = 100
eps = 10^-3
quiet = False

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU)")

# print(os.getcwd())
dict = torch.load('UAT Code/state1.pt')
model = dict['model']
train_data = dict['train_data']
train_targets = dict['train_targets']
test_data = dict['test_data']
test_targets = dict['test_targets']
weights = dict['state_dict']

adjusted_weights = copy.deepcopy(weights)
flattened_weights, num_dims = flatten_weights(weights, device)
loss_fn = torch.nn.MSELoss()  # mean square error

model.load_state_dict(weights)
model.eval()
y_pred = model(test_data)
mse = loss_fn(y_pred, test_targets)
mse = float(mse)
mse_array = [mse]
length_array = [0]
direction_array = [torch.zeros(num_dims)]
vector_array = [torch.zeros(num_dims).to(device)]
A = torch.randn(num_dims, p) * std
A = A.to(device)
A_pinv = torch.pinverse(A)
C_e_max = eps*torch.abs(A_pinv @ flattened_weights + 1)
C_e_min = -eps*torch.abs(A_pinv @ flattened_weights + 1)
highest_mse = -np.inf
for i in range(num_directions): 
    rand_tensor = 2 * torch.rand(num_dims) - 1
    rand_tensor = rand_tensor / torch.norm(rand_tensor)
    gaussian_lengths = torch.randn(num_lengths) * std
    for j in range(num_lengths):
        current_length = float(gaussian_lengths[j])
        new_length = rand_tensor * current_length
        new_length = new_length.to(device)
        new_weights_flattened = flattened_weights + new_length
        adjusted_weights = reshape_flattened_weights(new_weights_flattened, weights)
        model.load_state_dict(adjusted_weights)
        model.eval()
        y_pred = model(test_data)
        mse = loss_fn(y_pred, test_targets)
        mse = float(mse)
        mse_array.append(mse)
        length_array.append(current_length)
        if mse > highest_mse:
            highest_mse = mse
            highest_weights = copy.deepcopy(adjusted_weights)
        direction_array.append(rand_tensor)
        vector_array.append(new_length)
df = pd.DataFrame({"Loss": mse_array, "Length": length_array, "Direction": direction_array})
mse_array = np.array(mse_array)
vector_array = np.array(torch.stack(vector_array, dim=0).cpu())
if not quiet:
    plot_landscape(mse_array, vector_array)