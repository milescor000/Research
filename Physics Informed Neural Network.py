# Physics Informed Neural Network
# Corey Miles

'''
Create a PINN to capture the behavior of Burger's Equation, which is a nonlinear differential partial equation that describes the interaction between nonlinear
convection and linear diffusion.

Latin Hypercube: a method for generating a set of nearly uniformly distributed sample points in a multidimensional space.
'''

#---Libraries------------------------------------------------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import scipy as sc
from scipy.stats import qmc
import matplotlib.pyplot as plt
import time
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Precision------------------------------------------------------------------------------------------------------------------------------------------------------#
dtype = torch.float64  # sets the precision of tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # runs the simulation from the gpu if available, if not, uses the cpu
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Constraints/Parameters-----------------------------------------------------------------------------------------------------------------------------------------#
viscosity = 0.01/np.pi  #  viscosity of the flowing fluid
xmin, xmax = -1.0, 1.0  # one-dimensional (x) constraints
tmin, tmax = 0.0, 1.0  # time span
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Neural Network-------------------------------------------------------------------------------------------------------------------------------------------------#
class PINN(nn.Module):  # instantiate physics informed neural network with the nn.Module parent class

    def __init__(self, layers):  # constructor method
        super().__init__()  # constructor method of the parent class
        self.layers = nn.ModuleList()  # container that holds a list of activation functions
        for i in range(len(layers) - 1):  # iterate through layers
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))  # creates a linear activation function between each layer
        for m in self.layers:  # iterate through layers
            nn.init.xavier_normal_(m.weight)  # randomly initializes weights with variance across layers being roughly the same
            nn.init.zeros_(m.bias)  # initializes the biases of each layer as zero
        self.activation = torch.tanh  # stores the tanh funciton in an attribute

    def forward(self, t, x):  # forward function to execute the layers
        z = torch.cat([t, x], dim=1)  # combines the t and x tensors into one input tensor
        for i, layer in enumerate(self.layers[:-1]):  # iterates through stored function in each layer except for the last one
            z = layer(z)  #  performs linear operation on t,x tensor
            z = self.activation(z)  # performs tanh operation on t,x tensor
        z = self.layers[-1](z)  # performs the final linear operation on t,x tensor
        return z  # returns the output of the neural network
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Physics Informed Error-----------------------------------------------------------------------------------------------------------------------------------------#
def pde_error(model, t, x):  # defines function to calculate error based on physics equation
    u = model(t, x)  # calculate the output
    u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # differentiate u with respect to t
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # differentiate u with respect to x
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # differentiate u_x with respect to x
    f = u_t + u*u_x - viscosity*u_xx  # calculates error
    return f  # return error
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Data-----------------------------------------------------------------------------------------------------------------------------------------------------------#
Nu = 100  # number of conditions
Nf = 10000  # number of collocation points

N_init = Nu//2  # split number of conditions into two different parts
x_init = np.random.uniform(xmin, xmax, (N_init, 1))  # initialize random space coordinates
t_init = np.zeros_like(x_init)  # initialize starting time values
u_init = -np.sin(np.pi*x_init)  # initalizes values at boundary where t=0

N_b = Nu - N_init  # number of random boundary conditions
t_b = np.random.uniform(tmin, tmax, (N_b, 1))  # initialize random time stamps
x_b = np.where(np.random.rand(N_b, 1)<0.5, xmin, xmax)  # assigns approximately half of each to each boundary
u_b = np.zeros_like(x_b)  # initialuzes values at boundaries x = -1 or 1

t_u = np.vstack([t_init, t_b])  # combine all t values
x_u = np.vstack([x_init, x_b])  # combine all x values
u_u = np.vstack([u_init, u_b])  # combine all u values

t_u_t = torch.tensor(t_u, dtype=dtype, device=device, requires_grad=False)  # convert t values to tensor
x_u_t = torch.tensor(x_u, dtype=dtype, device=device, requires_grad=False)  # convert x values to tensor
u_u_t = torch.tensor(u_u, dtype=dtype, device=device, requires_grad=False)  # convert u values to tensor

sampler = qmc.LatinHypercube(d=2, seed=0)  # creates a latin hypercube sampler in a 2-dimensional space
sam = sampler.random(Nf)  # generates sample points

t_f = tmin + (tmax - tmin) * sam[:,0:1]  # map hypercube to time collocation points
x_f = xmin + (xmax - xmin) * sam[:,1:2]  # mpa hypercube to x collocation points

t_f_t = torch.tensor(t_f, dtype=dtype, device=device, requires_grad=True)  # convert t collocation points to tensor
x_f_t = torch.tensor(x_f, dtype=dtype, device=device, requires_grad=True)  # convert x collocation points to tensor
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Model Instantiation--------------------------------------------------------------------------------------------------------------------------------------------#
layers = [2] + [20]*9 + [1]  # creates a two input, 9 hidden layer, one output layer pattern
model = PINN(layers).to(device).to(dtype)  # instantiates the neural network
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Training-------------------------------------------------------------------------------------------------------------------------------------------------------#
lr = 0.001  # learning rate
epochs = 12000  # number of test epochs
print_every = 500  # points at which to print

optimizer = optim.Adam(model.parameters(), lr=lr)  # set mechanism to optimize weights

mse_loss = nn.MSELoss()  # mean squared error loss function

start_time = time.time()  # returns current time

for epoch in range(1, epochs+1):  # training iteration for each epoch
    model.train()  # prepares layers for training
    optimizer.zero_grad()  # resets gradients on each iteration

    t_u_train = t_u_t.clone().detach().requires_grad_(False)  # creates copy of t_u tensor
    x_u_train = x_u_t.clone().detach().requires_grad_(False)  # creates copy of x_u tenstor
    u_pred = model(t_u_train, x_u_train)  # outputs the predictions
    loss_u = mse_loss(u_pred, u_u_t)  # compares predictions to known u values

    t_f_train = t_f_t  # t collocation points used to train
    x_f_train = x_f_t  # x collocation points used to train
    f_pred = pde_error(model, t_f_train, x_f_train)  # physics prediction
    loss_f = mse_loss(f_pred, torch.zeros_like(f_pred))  # compares predictions to 0

    loss = loss_u + loss_f  # calculates total loss

    loss.backward()  # determines how much each layer affected loss
    optimizer.step()  # optimizer takes a step

    if epoch % print_every == 0 or epoch == 1:  # gives updates on progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:6d} | loss {loss.item():.3e} | loss_u {loss_u.item():.3e} | loss_f {loss_f.item():.3e} | time {elapsed:.1f}s")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---Plotting-------------------------------------------------------------------------------------------------------------------------------------------------------#
model.eval()  # prepares layers for estimation
Nt_plot = 200  # discretizes t-plot
Nx_plot = 200  # discretizes x-plot
t_plot = np.linspace(tmin, tmax, Nt_plot)  # creates time span
x_plot = np.linspace(xmin, xmax, Nx_plot)  # creates x span
T_grid, X_grid = np.meshgrid(t_plot, x_plot, indexing='xy')  # creates 2-dimension grid
t_in = torch.tensor(T_grid.flatten()[:,None], dtype=dtype, device=device, requires_grad=False)  # turns T_grid into an input tensor
x_in = torch.tensor(X_grid.flatten()[:,None], dtype=dtype, device=device, requires_grad=False)  # turns X_grid into an input tensor
with torch.no_grad():  # tells neural network to stop tracking gradients
    u_pred_grid = model(t_in, x_in).cpu().numpy().reshape(X_grid.shape)  # turns u predictions into a shape the same as the x_grid

fig = plt.figure(figsize=(10,4.5))  # create figure
ax1 = fig.add_axes([0.06, 0.55, 0.88, 0.38])  # sets location for plot
c = ax1.contourf(np.transpose(T_grid), np.transpose(X_grid), np.transpose(u_pred_grid), 50, cmap='RdBu_r') #  create contour map
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('u(t,x) — PINN prediction')  # set labels
plt.colorbar(c, ax=ax1, fraction=0.046, pad=0.04)  # set colorbar key

t_slice = 0.75  # location of time slice
x_slice = np.linspace(xmin, xmax, 300)[:,None]  # equally spaced x values
t_slice_arr = np.full_like(x_slice, t_slice)  # array same size as x_slice, but filled with t_slice
t_slice_t = torch.tensor(t_slice_arr, dtype=dtype, device=device, requires_grad=False)  # turns t_slice into a tensor
x_slice_t = torch.tensor(x_slice, dtype=dtype, device=device, requires_grad=False)  # turns x_slice into a tensor
with torch.no_grad():  # tells neural network to stop tracking gradients
    u_slice_pred = model(t_slice_t, x_slice_t).cpu().numpy().flatten()  # predicts slide outputs

ax2 = fig.add_axes([0.58, 0.08, 0.36, 0.36])  # small subplot on right bottom
ax2.plot(x_slice, u_slice_pred, 'r-', linewidth=2, label='Prediction')  # plots prediction versus x
ax2.set_xlabel('x'); ax2.set_ylabel('u(t,x)'); ax2.set_title(f't = {t_slice:.2f}')  # sets labels
ax2.legend()  # makes legend appear

plt.suptitle("PINN solution of Burgers' equation (Nu=100, Nf=10000)", fontsize=12)  # overall title
plt.show()  # makes plots appear
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

