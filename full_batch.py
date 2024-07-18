import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'Using device: {device}')
# lam = 0.4  


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.activation = torch.sin
        
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = 0.1 * out + identity
        return out

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layer1 = ResidualBlock(1, 24).double()
        self.hidden_layer2 = ResidualBlock(24, 17).double()
        self.hidden_layer3 = ResidualBlock(17, 10).double()
        self.hidden_layer4 = ResidualBlock(10, 3).double()
        self.hidden_layer5= ResidualBlock(3, 3).double()
        self.output_layer = ResidualBlock(3, 1).double()
        
    def forward(self, y):
        y = self.hidden_layer1(y)
        y = self.hidden_layer2(y)
        y = self.hidden_layer3(y)
        y = self.hidden_layer4(y)
        y = self.hidden_layer5(y)
        y = self.output_layer(y)
        return y
    
    def U(self, y):
        return (self(y) - self(-y)) / 2

    def get_lam(self):
        y = torch.linspace(-2,2,100,dtype=torch.float64).view(-1,1).to(device)
        y.requires_grad = True
        U = self.U(y)
        U_y = torch.autograd.grad(U, y, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        U_yy = torch.autograd.grad(U_y, y, grad_outputs=torch.ones_like(U_y), create_graph=True)[0]
        return torch.mean(torch.divide(-(1 + U_y) * U_y - (U + y)*U_yy, y*U_yy))
    
    def get_fixed_lam(self):
        return .4
    
def f(y,U,U_y,lam):
    return -lam * U + ((1 + lam) * y + U) * U_y

def compute_derivative(f, y, model, lam, orders,finite=False):
    y.requires_grad = True
    U = model.U(y)
    U_y = torch.autograd.grad(U, y, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    lam = model.get_lam()
    f_val = f(y, U, U_y, lam)
    h = y[1] - y[0]
    res = []
    if not finite:
        for _ in range(int(orders.max())):
            f_val = torch.autograd.grad(f_val, y, grad_outputs=torch.ones_like(f_val), create_graph=True)[0]
            if _ + 1 in orders:
                res.append(f_val)
    else:
        for _ in range(int(orders.max())):
            f_val = (y[1:] - y[:-1]) / h
            if _ + 1 in orders:
                res.append(f_val)
    return res


def Loss(model, y, collocation_points,mode,step):
    y.requires_grad = True
    U = model.U(y)
    U_y = torch.autograd.grad(U, y, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    U_yy = torch.autograd.grad(U, y, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    if mode == 'fixed':
        lam = model.get_fixed_lam()
    if mode == 'learned':
        lam = model.get_lam()


    # Equation loss
    f_val = f(y, U, U_y,lam)

    # Smooth loss 3rd and fifth derivative
    derivatives = compute_derivative(f,collocation_points,model,lam, orders=np.array([3.0]),finite=True)
    f_yyy = derivatives[0]
    # f_yyyyy = derivatives[1]
 

    # Condition loss U(-2) = 1
    g = model.U(torch.tensor([-2.0], dtype=y.dtype, device=y.device)) - 1
    
    equation_loss = torch.mean(f_val**2)
    condition_loss = torch.mean(g**2)

    experiment_loss = torch.exp(torch.tensor(data=[-0.1],dtype=torch.float64) * step) * torch.mean(U_yy**2)
    total_loss = equation_loss + condition_loss + experiment_loss + 1e-3*torch.mean(f_yyy**2) #+ 1e-7*torch.mean(f_yyyyy**2) 
    return total_loss

compiled_loss = torch.compile(Loss)
model = PINN().to(device)
model = torch.compile(model)
optimizer = optim.LBFGS(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

'''
 Load the model if needed
'''
# model.load_state_dict(torch.load('model.pth'))
# optimizer.load_state_dict(torch.load('optimizer.pth'))
# model.eval()  
# model.train()  

num_epochs = 1000
y_data = torch.linspace(1,2,1000,dtype=torch.float64).view(-1,1).to(device)
# y_data = (y_data-y_data.mean()) / y_data.std()

# writer = SummaryWriter()
Ns = 100
collocation_points = torch.FloatTensor(Ns).uniform_(-1, 1).view(-1, 1).double().to(device)
collocation_points = (collocation_points-collocation_points.mean()) / collocation_points.std()

def closure(step):
    optimizer.zero_grad() # Clear the gradients
    loss = compiled_loss(model, y_data,collocation_points,'fixed',step) # Compute the loss
    loss.backward() # Backward pass
    return loss

for epoch in range(num_epochs):
    # y_data = torch.FloatTensor(10000).uniform_(-2, 2).view(-1, 1).to(device)

    optimizer.zero_grad()
    loss = compiled_loss(model, y_data, collocation_points,'fixed',epoch)
    loss.backward()
    optimizer.step(lambda:closure(epoch))
    if epoch % 100 == 0:
        print(f'epoch {epoch} loss {loss.item()}') 
    # writer.add_scalar(f'Training Loss_1', loss.item(), epoch)
    if loss.item() <= 1e-8:
        break