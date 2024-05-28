import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def flatten_weights(weights, device):
    num_dims = 0
    flattened_weights = torch.empty(0)
    flattened_weights = flattened_weights.to(device)
    for key in weights:
        num_dims += torch.numel(weights[key])
        flatened_tensor = torch.flatten(weights[key])
        flattened_weights = torch.cat((flattened_weights, flatened_tensor))
    return flattened_weights, num_dims

def reshape_flattened_weights(flat_weights, original_weights):
    start_index = 0
    adjusted_weights = {}
    for key in original_weights:
        tensors = original_weights[key]
        desired_shape = tensors.size()
        num_elems = tensors.numel()
        end_index = start_index + num_elems
        new_weight = flat_weights[start_index:end_index]
        start_index = end_index
        new_weight = new_weight.reshape(desired_shape)
        adjusted_weights[key] = new_weight
    return adjusted_weights

def plot_landscape(mse_array, vector_array):
    U, S, Vt = np.linalg.svd(vector_array)
    U = U[:,:2]
    S = S[:2]

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
    
    # sharpness = (highest_mse - mse)/(1 + mse) * 100
    # print(sharpness)
    
    # show plot
    plt.show()