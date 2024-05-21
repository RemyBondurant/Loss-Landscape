import copy
import torch
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

num_lengths = 100
num_directions = 10
std = 1
quiet = False

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU)")

model = torch.load('test_model.pt')
test_targets = torch.load('test_targets.pt')
test_data = torch.load('test_data.pt')
weights = torch.load('test_state_dict.pt', weights_only=True)
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
mse_array = [mse]
length_array = [0]
direction_array = [torch.zeros(num_dims)]
vector_array = [torch.zeros(num_dims).to(device)]
for i in range(num_directions): 
    rand_tensor = 2 * torch.rand(num_dims) - 1
    rand_tensor = rand_tensor / torch.norm(rand_tensor)
    # print(torch.norm(rand_tensor))
    gaussian_lengths = torch.randn(num_lengths) * std
    for j in range(num_lengths):
        current_length = float(gaussian_lengths[j])
        new_length = rand_tensor * current_length
        new_length = new_length.to(device)
        new_weights_flattened = flattened_weights + new_length
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
        mse = loss_fn(y_pred, test_targets)
        mse = float(mse)
        mse_array.append(mse)
        length_array.append(current_length)
        # print("RAND TENSOR")
        # print(rand_tensor)
        direction_array.append(rand_tensor)
        vector_array.append(new_length)
        # if not quiet:
        #     print("NEW MSE: " + str(mse))
# for key in weights:
#     print(torch.equal(weights[key], adjusted_weights[key]))
df = pd.DataFrame({"Loss": mse_array, "Length": length_array, "Direction": direction_array})
mse_array = np.array(mse_array)
vector_array = np.array(torch.stack(vector_array, dim=0).cpu())
if not quiet:
    U, S, Vt = np.linalg.svd(vector_array)
    U = U[:,:2]
    S = S[:2]
    reduced = U @ np.diag(S)

    # Creating dataset
    x = U[:,0]
    y = U[:,1]
    z = mse_array
    
    # Creating figure
    fig = plt.figure(figsize =(16, 9))  
    ax = plt.axes(projection ='3d')  
    
    # Creating color map
    my_cmap = plt.get_cmap('hot')
    
    # Creating plot
    trisurf = ax.plot_trisurf(x, y, z,
                            cmap = my_cmap,
                            linewidth = 0.2, 
                            antialiased = True,
                            edgecolor = 'grey')  
    fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)
    ax.set_title('SVD Loss Plot')
    
    # Adding labels
    ax.set_xlabel('Component 1', fontweight ='bold') 
    ax.set_ylabel('Component 2', fontweight ='bold') 
    ax.set_zlabel('Loss Value', fontweight ='bold')
        
    # show plot
    plt.show()