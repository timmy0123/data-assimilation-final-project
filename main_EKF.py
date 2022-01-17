import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self,input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim,input_dim*7)
        self.layer2 = nn.Linear(input_dim*7,input_dim*15)
        self.layer3 = nn.Linear(input_dim*15,input_dim*20)
        self.layer4 = nn.Linear(input_dim*20,input_dim*20)
        self.layer5 = nn.Linear(input_dim*20,input_dim*15)
        self.layer6 = nn.Linear(input_dim*15,40)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        x = self.layer4(x)
        x = torch.tanh(x)
        x = self.layer5(x)
        x = torch.tanh(x)
        x = self.layer6(x)
        return x


def l96_jac(y, dt):
    n = np.shape(y)[0]
    J = np.zeros([n,n])
    for j in range(n-1):
        J[j,j-2] = -y[j-1]
        J[j,j-1] = y[j+1] - y[j-2]
        J[j,j]   = -1
        J[j,j+1] = y[j-1]
    J[n-1,n-3] = -y[n-2]
    J[n-1,n-2] = y[0] - y[n-3]
    J[n-1,n-1] = -1
    J[n-1,0]   = y[n-2]
    return dt*J + np.eye(n)


def EKF_assimilation(x,Pb,tt,y_o,err_all):
  tts = tt - 1
  Ts = tts * dT  # forecast start time
  Ta = tt  * dT  # forecast end time (DA analysis time)

  solver = ode(lorenz96.f).set_integrator('dopri5')
  solver.set_initial_value(x, Ts).set_f_params(F)
  solver.integrate(Ta)
  x_b = solver.y
  jacF = l96_jac(x,dT)
  Pb = jacF.dot(Pb).dot(jacF.T)
  Pb = .5*(Pb + Pb.T)
  x_b = x_b.transpose()

  #y_o = np.genfromtxt('train_data_/x_t2.txt')
  #err_all = np.genfromtxt('train_data_/err.txt')
  a = err_all[(tt-1):tt,:]
  y_o_err = y_o[tt,range(0,40,2)] + a
  R = np.eye(20)
  H = np.zeros((20,40),dtype=float)
  for row in range(H.shape[0]):
    H[row,row*2] = 1
  y_b = np.dot(H, x_b)
  d = y_o_err.T - y_b[:,np.newaxis]


  K = np.dot(np.dot(Pb,H.T),np.linalg.inv(np.dot(np.dot(H,Pb),H.T) + R))
  Pa = np.dot((np.eye(40) - np.dot(K,H)),Pb)
  x_a = x_b[:,np.newaxis] + np.dot(K,d)

  return x_a,(Pa + 0.5)


model = Model(80)
model.train()
model.cuda()
lossop = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1E-5)
predx = np.genfromtxt('train_data_/x_a_init.txt')
predy = np.genfromtxt('train_data_/x_t2.txt')

out1 = [predx]
err = np.genfromtxt('train_data_/err.txt')
save_time = [50,100,500,750,1000]
pbar = tqdm(range(1001))
lossout = []
for _ in pbar:
  lossess = []
  out = [predx]
  assimilation_out = [predx]
  Pb = np.eye(40)
  for n in range(1,(predy.shape[0])):
      ass,Pb = EKF_assimilation(assimilation_out[-1],Pb,n,predy,err)
      ass = np.squeeze(ass,axis=1)
      assimilation_out.append(ass)
      pred = model(torch.Tensor(np.append(out[-1],ass)).cuda())
      out.append(pred.cpu().detach().numpy())
      optimizer.zero_grad()
      loss = lossop(pred,torch.Tensor(predy[n,:]).cuda())
      lossess.append(loss.item())
      loss.backward()
      optimizer.step()
  pbar.set_postfix({'losses': sum(lossess)/200})
  lossout.append(sum(lossess)/200)
  if _ in save_time:
    savename = "Model_with_EKF/model{:.0f}.pt".format((_))
    torch.save(model,savename)