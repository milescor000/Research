# Basic Re-Entry CNN
# Corey Miles

#---Notes------------------------------------------------------------------------------------------------------------------------------------------------------------#

# next steps:
# residual, naive system
# CNN upscaling

# Ideas:
# patch-wise generation

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Libraries--------------------------------------------------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
import torch.optim as optim
import time
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Global Variables-------------------------------------------------------------------------------------------------------------------------------------------------#
resolution = 64                                                      # grid resolution dimension
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Low-Fidelity Data------------------------------------------------------------------------------------------------------------------------------------------------#
lf_data = pd.read_csv("LowFidelityData.csv")                         # read-in low-fidelity data
lf_u = lf_data["Velocity[i] (m/s)"]                                  # unpack velocity in the x-direction (u)
lf_x = lf_data["X (m)"]                                              # unpack x-coordinates
lf_y = lf_data["Y (m)"]                                              # unpack y-coordinates

lf_u = np.array([lf_u])                                              # convert u-velocity into an array
lf_x = np.array([lf_x])                                              # convert x-coordinates into an arraywe3
lf_y = np.array([lf_y])                                              # convert y-coordinates into an array

lfx_res = np.linspace(lf_x.min(), lf_x.max(), resolution)            # grid x-resolution
lfy_res = np.linspace(lf_y.min(), lf_y.max(), resolution)            # grid y-resolution

lfx_res, lfy_res = np.meshgrid(lfx_res, lfy_res)                     # create resolution x resolution grid

lf_ui = griddata((lf_x.squeeze(), lf_y.squeeze()), lf_u.squeeze(),   # interpolate u-velocity data
                 (lfx_res, lfy_res), method="cubic")

mask = ~np.isnan(lf_ui)                                              # create a mask of valid values

if np.any(~mask):                                                    # check for any NaN values

    valid_coords = np.array(np.where(mask)).T                        # provides coordinates of valid values
    valid_values = lf_ui[mask]                                       # produce map for valid values

    fill_interpolator = NearestNDInterpolator(valid_coords,          # lookup tool for finding nearest value
                                              valid_values)
    
    missing_coords = np.array(np.where(~mask)).T                     # provides coordinates of NaN values

    filled_values = fill_interpolator(missing_coords)                # creates a nearest value to each NaN value
    lf_ui[~mask] = filled_values                                     # replaces NaN values with nearest value
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---High-Fidelity Data-----------------------------------------------------------------------------------------------------------------------------------------------#
hf_data = pd.read_csv("HighFidelityData.csv")                        # read-in high-fidelity data
hf_u = hf_data["Velocity[i] (m/s)"]                                  # unpack velocity in the x-direction (u)
hf_x = hf_data["X (m)"]                                              # unpack x-coordinates
hf_y = hf_data["Y (m)"]                                              # unpack y-coordinates

hf_u = np.array([hf_u])                                              # convert u-velocity into an array
hf_x = np.array([hf_x])                                              # convert x-coordinates into an array
hf_y = np.array([hf_y])                                              # convert y-coordinates into an array

hfx_res = np.linspace(hf_x.min(), hf_x.max(), resolution)            # grid x-resolution
hfy_res = np.linspace(hf_y.min(), hf_y.max(), resolution)            # grid y-resolution

hfx_res, hfy_res = np.meshgrid(hfx_res, hfy_res)                     # create resolution x resolution grid

hf_ui = griddata((hf_x.squeeze(), hf_y.squeeze()), hf_u.squeeze(),   # interpolate u-velocity data
                 (hfx_res, hfy_res), method="cubic")

mask = ~np.isnan(hf_ui)                                              # create a mask of valid values

if np.any(~mask):                                                    # check for any NaN values

    valid_coords = np.array(np.where(mask)).T                        # provides coordinates of valid values
    valid_values = hf_ui[mask]                                       # produce map for valid values

    fill_interpolator = NearestNDInterpolator(valid_coords,          # lookup tool for finding nearest value
                                              valid_values)
    
    missing_coords = np.array(np.where(~mask)).T                     # provides coordinates of NaN values

    filled_values = fill_interpolator(missing_coords)                # creates a nearest value to each NaN value
    hf_ui[~mask] = filled_values                                     # replaces NaN values with nearest value
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Normalization----------------------------------------------------------------------------------------------------------------------------------------------------#
global_max_u = max(np.max(lf_ui), np.max(hf_ui))                     # calculate global max u-velocity

lfu_norm = lf_ui/global_max_u                                        # normalize low-fidelity data
hfu_norm = hf_ui/global_max_u                                        # normalize high-fidelity data

lfu_tensor = torch.tensor(                                           # turn into tensor
    lfu_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

hfu_tensor = torch.tensor(                                           # turn into tensor
    hfu_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Convolutional Neural Network-------------------------------------------------------------------------------------------------------------------------------------#
class CNN(nn.Module):                                                # define CNN class
    def __init__(self, in_channels=1, out_channels=1,                # define layer parameters
                 hidden=[64, 64, 64, 64, 64]):
        super().__init__()                                           # define CNN subclass

        self.conv1 = nn.Conv2d(in_channels, hidden[0],               # layer 1 convolution
                               kernel_size=3, padding=1)
        self.act1 = nn.ReLU()                                        # layer 1 activation

        self.conv2 = nn.Conv2d(hidden[0], hidden[1],                 # layer 2 convolution
                               kernel_size=3, padding=1)
        self.act2 = nn.ReLU()                                        # layer 2 activation

        self.conv3 = nn.Conv2d(hidden[1], hidden[2],                 # layer 3 convolution
                               kernel_size=3, padding=1)
        self.act3 = nn.ReLU()                                        # layer 3 activation

        self.conv4 = nn.Conv2d(hidden[2], hidden[3],                 # layer 4 convolution
                               kernel_size=3, padding=1)
        self.act4 = nn.ReLU()                                        # layer 4 activation

        self.conv5 = nn.Conv2d(hidden[3], hidden[4],                 # layer 5 convolution
                               kernel_size=3, padding=1)
        self.act5 = nn.ReLU()                                        # layer 5 activation

        self.conv_out = nn.Conv2d(hidden[4], out_channels,           # output layer
                               kernel_size=3, padding=1)
        
        for m in self.modules():                                     # run through self functions
            if isinstance(m, nn.Conv2d):                             # check if self function is a convolution
                nn.init.kaiming_uniform_(m.weight,                   # set weights
                                         nonlinearity="relu")
                if m.bias is not None:                               # check biases
                    nn.init.zeros_(m.bias)                           # set biases to 0

    def forward(self, x):                                            # define CNN forward pass
        x = self.act1(self.conv1(x))                                 # run layer 1
        x = self.act2(self.conv2(x))                                 # run layer 2
        x = self.act3(self.conv3(x))                                 # run layer 3
        x = self.act4(self.conv4(x))                                 # run layer 4
        x = self.act5(self.conv5(x))                                 # run layer 5
        out = self.conv_out(x)                                       # CNN output
        return out                                                   # return CNN output
    
model = CNN()                                                        # set variable to call CNN
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Training---------------------------------------------------------------------------------------------------------------------------------------------------------#
optimizer = optim.Adam(model.parameters(), lr=1e-3)                  # set optimizer type as adam
scheduler = optim.lr_scheduler.StepLR(optimizer,                     # initialize scheduler
                                      step_size=200, gamma=0.5)
mse_loss = nn.MSELoss()                                              # set loss function as nean squared error
epochs = 1000                                                        # number of training epochs
print_epochs = 100                                                   # number of timing epochs
epoch_array = np.arange(1, epochs+1)                                 # store epochs as an array
start_time = time.time()                                             # start timer

loss_array = np.zeros(epochs)                                        # initialize empty loss array

for epoch in range(epochs):                                          # iterate through epochs

    model.train()                                                    # begin training setting
    optimizer.zero_grad()                                            # reset gradient for each iteration

    pred = model(lfu_tensor)                                         # run normalized data through the neural network

    loss = mse_loss(pred + lfu_tensor, hfu_tensor)                   # calculate loss (residual)
    loss_array[epoch] = loss.item()                                  # append loss to loss array

    loss.backward()                                                  # calculate backwards gradient

    optimizer.step()                                                 # move optimizer in the correct direction
    scheduler.step()                                                 # update scheduler 

    if epoch % print_epochs == 0 or epoch == 1:                      # check for timing epoch
        elapsed = time.time() - start_time                           # calculate elapsed time
        print(f"Epoch {epoch:6d} | loss {loss.item():.3e}; | "       # print loss at timing epoch
              f"time {elapsed:.1f}s")
        
total_training_time = time.time() - start_time                       # calculate total training time
print(f"\nTotal training time: {total_training_time:.2f} seconds")   # print total training time
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Testing----------------------------------------------------------------------------------------------------------------------------------------------------------#
model.eval()                                                         # begin evaluation setting

with torch.no_grad():                                                # turn off gradient tracking
    
    norm_pred = model(lfu_tensor).squeeze()                          # run normalized data through the neural network
    pred = (norm_pred*global_max_u).numpy()                          # unnormalized data and convert to an array
    final_pred = pred + (lfu_tensor.squeeze().numpy()*global_max_u)  # add residual back to the low-fidelity data
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Post-Processing--------------------------------------------------------------------------------------------------------------------------------------------------#
hfu_norm_array = hfu_tensor.squeeze().numpy()                        # convert high-fidelity tensor to an array
hfu_array = hfu_norm_array*global_max_u                              # unnormalized high-fidelity array
error_per = abs((hfu_array - final_pred)/hfu_array)*100              # calculate percent error
print(f"Maximum Percent Error: {round(np.max(error_per), 2)}%")      # print the maximum percent error
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Figures----------------------------------------------------------------------------------------------------------------------------------------------------------#
plt.figure()                                                         # figure 1: low-fidelity data
plt.title("Low-Fidelity Data")                                       # name plot
plt.pcolormesh(lfx_res, lfy_res, lf_ui, cmap=cm.coolwarm,            # plot heat map
               vmin=lf_u.min(), vmax=lf_u.max())
plt.colorbar()                                                       # add colorbar key

plt.figure()                                                         # figure 2: high-fidelity data
plt.title("High-Fidelity Data")                                      # name plot
plt.pcolormesh(hfx_res, hfy_res, hf_ui, cmap=cm.coolwarm,            # plot heat map
               vmin=hf_u.min(), vmax=hf_u.max())
plt.colorbar()                                                       # add colorbar key

plt.figure()                                                         # figure 3: prediction heat map
plt.title("CNN Prediction")                                          # name plot
plt.pcolormesh(hfx_res, hfy_res, final_pred, cmap=cm.coolwarm,       # plot heat map
               vmin=hf_u.min(), vmax=hf_u.max())
plt.colorbar()                                                       # add colorbar key

plt.figure()                                                         # figure 4: percent error heat map
plt.title(f"% error")                                                # name plot
plt.pcolormesh(hfx_res, hfy_res, error_per, cmap=cm.YlGn,            # plot heat map
               vmin=0, vmax=18)
plt.colorbar()                                                       # add colorbar key

plt.figure()                                                         # figure 5: loss graph
plt.title(f"Loss")                                                   # name plot
plt.plot(epoch_array, loss_array)                                    # plot graph

plt.show()                                                           # show all figures
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#