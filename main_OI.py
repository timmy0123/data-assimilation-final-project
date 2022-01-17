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


def OI_assimilation(x,tt,y_o,err_all):
  tts = tt - 1
  Ts = tts * dT  # forecast start time
  Ta = tt  * dT  # forecast end time (DA analysis time)

  solver = ode(lorenz96.f).set_integrator('dopri5')
  solver.set_initial_value(x, Ts).set_f_params(F)
  solver.integrate(Ta)
  x_b = solver.y
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

  Pb = np.eye(40)
  K = np.dot(np.dot(Pb,H.T),np.linalg.inv(np.dot(np.dot(H,Pb),H.T) + R))
  x_a = x_b[:,np.newaxis] + np.dot(K,d)

  return x_a


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
  for n in range(1,(predy.shape[1] - 1)):
    temp_ass = []
    for i in range(assimilation_out[-1].shape[0]):
      err = np.random.normal(0,0.25,[200,20])
      ass = OI_assimilation(assimilation_out[-1][i,:],n,predy[i,:,:],err)
      ass = np.squeeze(ass,axis=1)
      temp_ass.append(ass)
    assimilation_out.append(np.array(temp_ass))
    optimizer.zero_grad()
    predx = model(torch.Tensor(np.append(out[-1], np.array(temp_ass),axis=1)).cuda())
    loss = lossop(predx,torch.Tensor(predy[:,n,:]).cuda())
    out.append(predx.cpu().detach().numpy())
    loss.backward()
    optimizer.step()
    lossess.append(loss.item())
  pbar.set_postfix({'losses': sum(lossess)/200})
  lossout.append(sum(lossess)/200)
  if _ in save_time:
    savename = "model_with_assimilation2/model{:.0f}_4.pt".format((_+300))
    torch.save(model,savename)
savename = "model_with_assimilation2/model{:.0f}_4.pt".format((_+300))
torch.save(model,savename)