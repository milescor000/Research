# Convolutional Neural Network
# Corey Miles

'''
Create a CNN to perform a super resolution of course MRI data.

Computational Domain: a set of coordinates that are uniform in the computational domain (xi, eta, zeta) but when transformed to cartesian coordinates (x, y, z)
becomes non-uniform and non-orthogonal.

Upsample: imakes a data representation larger, typically by inserting new points between existing ones. 

Bicubic: uses a 4x4 matrix to calculate gradients between points and estimate what pixels/points would look like between the existing pixels/points
'''

#---Libraries------------------------------------------------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Set-Up---------------------------------------------------------------------------------------------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # runs CNN through cpu unless a gpu is available
torch.manual_seed(0)  # creates reproducible random results
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Data-----------------------------------------------------------------------------------------------------------------------------------------------------------#
lfdata = np.load("sr_lfdata.npy")  # loads low-fidelity data
lfx = lfdata[0, :, :]  # x position (14x9)
lfy = lfdata[1, :, :]  # y position
lfu = lfdata[4, :, :]  # x velocity
lfv = lfdata[5, :, :]  # y velocity

plt.figure()  # initializes a figure
plt.pcolormesh(lfx, lfy, np.sqrt(lfu**2 + lfv**2), cmap=cm.coolwarm, vmin=0.0, vmax=1.0)  # creates a color mesh figure
plt.colorbar()  # creates a color bar to the side of the color mesh figure

hfdata = np.load("sr_hfdata.npy")  # loads high-fidelity data
Jinv = hfdata[0, :, :]  # inverse jacobian for scaling factor
dxdxi = hfdata[1, :, :]  # computational x positions with respect to xi
dxdeta = hfdata[2, :, :]  # computational x positions with respect to eta
dydxi = hfdata[3, :, :]  # computational y positions with respect to xi
dydeta = hfdata[4, :, :]  # computational y positions with respect to eta
hfx = hfdata[5, :, :]  # high-fidelity x positions
hfy = hfdata[6, :, :]  # high-fidelity y positions

ny, nx = hfx.shape  # grid size
h = 0.01  # high-fidelity grid spacing

u_lr = torch.tensor(lfu, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # x velocity tensor:(batch, channel, length)|(H, W)->(C, H, W)->(B, C, H, W)
v_lr = torch.tensor(lfv, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # y velocity tensor:(batch, channel, length)|(H, W)->(C, H, W)->(B, C, H, W)
input_lr = torch.cat([u_lr, v_lr], dim=1)  # combines u and v tensors into one tensor
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Upsampling-----------------------------------------------------------------------------------------------------------------------------------------------------#
upsample = nn.Upsample(size=(ny, nx), mode='bicubic', align_corners=True)  # resizes (H, W)->(ny, nx)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Convolutional Neural Network-----------------------------------------------------------------------------------------------------------------------------------#
class CNN(nn.Module):  # instantiate convolutional neural network with the nn.Module parent class
    def __init__(self, in_channels=2, out_channels=3, hidden=[16, 32, 16]):  # constructor method
        super().__init__()  # constructor method of the parent class

        self.conv1 = nn.Conv2d(in_channels, hidden[0], kernel_size=5, padding=2)  # input convolutional layer
        self.act1 = nn.ReLU()  # rectified linear unit activation function, introduces non-linearity

        self.conv2 = nn.Conv2d(hidden[0], hidden[1], kernel_size=5, padding=2)  # creates a convolutional layer
        self.act2 = nn.ReLU()  # rectified linear unit activation function, introduces non-linearity

        self.conv3 = nn.Conv2d(hidden[1], hidden[2], kernel_size=5, padding=2)  # creates a convolutional layer
        self.act3 = nn.ReLU() # rectified linear unit activation function, introduces non-linearity

        self.conv_out = nn.Conv2d(hidden[2], out_channels, kernel_size=5, padding=2) # output convolutional layer

        for m in self.modules():  # iterates through layers
            if isinstance(m, nn.Conv2d):  # checks to see if layer is a convolution
                nn.init.uniform_(m.weight, -1.0/np.sqrt(25*in_channels), 1.0/np.sqrt(25*in_channels))  # uniformly distributes weight tensor
                if m.bias is not None:  # checks to see if layer has a bias tensor
                    nn.init.zeros_(m.bias)  # initializes biases as 0

    def forward(self, x):  # forward function to execute the layers
        x = self.act1(self.conv1(x))  # first round of transformations
        x = self.act2(self.conv2(x))  # second round of transformations
        x = self.act3(self.conv3(x))  # third round of transformations
        out = self.conv_out(x)  # transforms to the output
        return out  # returns output
    
model = CNN().to(device)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Derivatives----------------------------------------------------------------------------------------------------------------------------------------------------#
def ddxi(f, h):  # gradients in the transformed x direction
    dfdx_central = (f[:, :, :, 0:-4] - 8*f[:, :, :, 1:-3] + 8*f[:, :, :, 3:-1] - f[:, :, :, 4:]) / (12*h)  # central differencing function
    dfdx_left = (-11*f[:, :, :, 0:2] + 18*f[:, :, :, 1:3] - 9*f[:, :, :, 2:4] + 2*f[:, :, :, 3:5]) / (6*h)  # left side differencing function
    dfdx_right = (-2*f[:, :, :, -5:-3] + 9*f[:, :, :, -4:-2] - 18*f[:, :, :, -3:-1] + 11*f[:, :, :, -2:]) / (6*h)  # right side differencing function
    return torch.cat((dfdx_left, dfdx_central, dfdx_right), dim=3)  # returns full list of gradients

def ddeta(f, h):  # gradients in the transformed y direction
    dfdy_central = (f[:, :, 0:-4, :] - 8*f[:, :, 1:-3, :] + 8*f[:, :, 3:-1, :] - f[:, :, 4:, :]) / (12*h)  # central differencing function
    dfdy_bot = (-11*f[:, :, 0:2, :] + 18*f[:, :, 1:3, :] -9*f[:, :, 2:4, :] + 2*f[:, :, 3:5, :]) / (6*h)  # bottom differencing function
    dfdy_top = (-2*f[:, :, -5:-3, :] + 9*f[:, :, -4:-2, :] -18*f[:, :, -3:-1, :] + 11*f[:, :, -2:, :]) / (6*h)  # top differencing function
    return torch.cat((dfdy_bot, dfdy_central, dfdy_top), dim=2)

Jinv_t   = torch.tensor(Jinv, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # convert inverse jacobian to tensor
dxdxi_t  = torch.tensor(dxdxi, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # convert computational x with respect to xi to tensor
dxdeta_t = torch.tensor(dxdeta, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # convert computational x with respect to eta to tensor
dydxi_t  = torch.tensor(dydxi, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # convert computational y with respect to xi to tensor
dydeta_t = torch.tensor(dydeta, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # convert computational y with respect to eta to tensor

def map_ref_to_phys(df_dxi, df_deta):  # function to map coordinates
    df_dx = Jinv_t * (df_dxi * dydeta_t - df_deta * dydxi_t)  # maps computational x to physical x coordinate
    df_dy = Jinv_t * (df_deta * dxdxi_t - df_dxi * dxdeta_t)  # maps computational y to physical y coordinate
    return df_dx, df_dy  # returns mappings
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Physics Error--------------------------------------------------------------------------------------------------------------------------------------------------#
rho = 1  # fluid density
nu = 0.01  # fluid viscosity

def pde_error(pred):  # function to calculate physics error

    u = pred[:, 0:1, :, :]  # x velocity field
    v = pred[:, 1:2, :, :]  # y velocity field
    p = pred[:, 2:3, :, :]  # pressure field

    u_xi = ddxi(u, h)  # u with respect to xi
    u_eta = ddeta(u, h)  # u with respect to eta
    v_xi = ddxi(v, h)  # v with respect to xi
    v_eta = ddeta(v, h)  # v with respect to eta
    p_xi = ddxi(p, h)  # p with respect to xi
    p_eta = ddeta(p, h)  # p with respect to eta

    dudx, dudy = map_ref_to_phys(u_xi, u_eta)  # map computational x to physical x
    dvdx, dvdy = map_ref_to_phys(v_xi, v_eta)  # map computational y to physical y
    dpdx, dpdy = map_ref_to_phys(p_xi, p_eta)  # map computational p to physical p

    dudx_xi = ddxi(dudx, h)  # dudx with respect to xi
    dudx_eta = ddeta(dudx, h)  # dudx with respect to eta
    d2udx2, _ = map_ref_to_phys(dudx_xi, dudx_eta)  # returns d2udx2 and d2udxdy

    dudy_xi = ddxi(dudy, h)  # dudy with respect to xi
    dudy_eta = ddeta(dudy, h)  # dudy with respect to eta
    _, d2udy2 = map_ref_to_phys(dudy_xi, dudy_eta)  # returns d2udydx and d2udy2

    dvdx_xi = ddxi(dvdx, h)  # dvdx with respect to xi
    dvdx_eta = ddeta(dvdx, h)  # dvdx with respect to eta
    d2vdx2, _ = map_ref_to_phys(dvdx_xi, dvdx_eta)  # returns d2uvdx2 and d2vdxdy

    dvdy_xi = ddxi(dvdy, h)  # dvdv with respect to xi
    dvdy_eta = ddeta(dvdy, h)  # dvdy with respect to eta
    _, d2vdy2 = map_ref_to_phys(dvdy_xi, dvdy_eta)  # returns d2vdydx and d2vdy2

    conv_x = u*dudx + v*dudy  # x convective acceleration
    conv_y = u*dvdx + v*dvdy  # y convective acceleration

    r_cont = dudx + dvdy  # continuity equation
    r_xmom  = conv_x + (1.0/rho) * dpdx - nu * (d2udx2 + d2udy2)  # x momentum equation
    r_ymom  = conv_y + (1.0/rho) * dpdy - nu * (d2vdx2 + d2vdy2)  # y momentum equation

    return r_cont, r_xmom, r_ymom
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Boundary Conditions--------------------------------------------------------------------------------------------------------------------------------------------#
def enforce_bcs(pred):  # enforce boundary conditions

    N, C, H, W = pred.shape  # get shape of prediction tensor
    top = H-1  # bottom row indice
    bottom = 0  # top row indice
    left = 0  # left column indice
    right = W-1  # right column indice

    #  bottom boundary conditions
    #  u=0, v=1, dpdeta=0
    pred[:, 0, bottom, :] = 0.00  # bottom u = 0
    pred[:, 1, bottom, :] = 1.0  # botom v = 1
    pred[:, 2, bottom, :] = pred[:, 2, bottom+1, :]  # bottom dp = 0

    #  top boundary conditions
    #  dudeta=0, dvdeta=0, p=0
    pred[:, 0, top, :] = pred[:, 0, top-1, :]  # top du = 0
    pred[:, 1, top, :] = pred[:, 1, top-1, :]  # top dv = 0
    pred[:, 2, top, :] = 0.0  # top p = 0

    #  left boundary conditions
    #  u=0, v=0, dpdxi=0
    pred[:, 0, :, left] = 0.0  # left u = 0
    pred[:, 1, :, left] = 0.0  # left v = 0
    pred[:, 2, :, left] = pred[:, 2, :, left+1]  # left dp = 0

    #  right boundary conditions
    #  u=0, v=0, dpdxi=0
    pred[:, 0, :, right] = 0.0  # right u = 0
    pred[:, 1, :, right] = 0.0  # right v = 0
    pred[:, 2, :, right] = pred[:, 2, :, right-1]  # right dp = 0

    return pred
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Training-------------------------------------------------------------------------------------------------------------------------------------------------------#
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # select optimizer and learning rate
mse_loss = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)  # decrease learning rate by half every 5000 epochs
epochs = 20000  # number of iterations
print_epochs = 2000  # passed iterations to print

w_cont = 1.0  # continuity weight
w_mom = 1e-4  # momentum weight

for epoch in range(epochs):  # iterate through epochs

    model.train()  # prepares layers for training
    optimizer.zero_grad()  # resets gradients

    up_in = upsample(input_lr)  # upsample to larger size
    raw_out = model(up_in)  # raw predictions
    pred = raw_out.clone()  # clone output
    pred = enforce_bcs(pred)  # enforce boundary conditions

    r_cont, r_xmom, r_ymom = pde_error(pred)  # physics errors of the prediction
    loss_cont = mse_loss(r_cont, torch.zeros_like(r_cont))  # continuity loss
    loss_xmom = mse_loss(r_xmom, torch.zeros_like(r_xmom))  # x momentum loss
    loss_ymom = mse_loss(r_ymom, torch.zeros_like(r_ymom))  # y momentum loss

    loss = loss_cont*w_cont + loss_xmom*w_mom + loss_ymom*w_mom  # total loss

    loss.backward()  # determines effect of each layer
    optimizer.step()  # steps in the direction of optimization
    scheduler.step()  # steps the scheduler in the direction of gamma if necessary

    if (epoch % print_epochs) == 0:
        print(f"Epoch {epoch:05d}  Loss {loss.item():.6e}  Cont Loss {loss_cont.item():.3e}  Xmom Loss {loss_xmom.item():.3e}  Ymom Loss {loss_ymom.item():.3e}")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Plot Predictions-----------------------------------------------------------------------------------------------------------------------------------------------#
model.eval()  # prepares layers for estimation
with torch.no_grad():  # gradients not tracked
    up_in = upsample(input_lr)  # upsample to larger size
    raw_out = model(up_in)  # raw predictions
    pred = enforce_bcs(raw_out)  # enforce boundary conditions
    u_pred = pred[0,0].cpu().numpy()  # unpack x velocity as array
    v_pred = pred[0,1].cpu().numpy()  # unpack y velocity as array

plt.figure()
plt.pcolormesh(hfx, hfy, np.sqrt(u_pred**2 + v_pred**2), cmap=cm.coolwarm, vmin=0.0, vmax=1.3)  # create color mesh figure
plt.colorbar()  # colorbar next to color mesh figure
plt.title("CNN-SR predicted speed (u,v)")  # plot title
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Show Plots-----------------------------------------------------------------------------------------------------------------------------------------------------#
plt.show()  # show all plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#