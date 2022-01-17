import numpy as np
from scipy.integrate import ode
import lorenz96
from main_OI import OI_assimilation
from settings import *
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


RMSE_ALL = []
Bias_ALL = []
ALL_pred = []
lossop = nn.MSELoss()
save_time = [500]
for n in save_time:
  load_pth = "Model_with_EKF/model{:.0f}.pt".format(n)
  model = torch.load(load_pth)
  model.eval()
  predx = np.genfromtxt('train_data_/x_a_init.txt')
  #predx = data.init[50]
  predy = np.genfromtxt('train_data_/x_t2.txt')
  #predy = data.ground_truth[50]
  out1 = [predx]
  assimilation_out = [predx]
  lossess = []
  err = np.genfromtxt('train_data_/err.txt')
  #err = np.random.normal(0,0.25,[200,20])
  Pb = np.eye(40)
  for n in range(1,(predy.shape[0])):
    ass = OI_assimilation(assimilation_out[-1],n,predy,err)
    ass = np.squeeze(ass,axis=1)
    assimilation_out.append(ass)
    pred = model(torch.Tensor(np.append(out1[-1],ass)).cuda())
    loss = lossop(pred,torch.Tensor(predy[n,:]).cuda())
    lossess.append(loss.item())
    out1.append(pred.cpu().detach().numpy())
  out1 = np.array(out1)
  RMSE = []
  Bias = []
  for n in range(out1.shape[0]):
    RMSE.append(np.sqrt(((out1[n,:] - predy[n,:]) ** 2).mean()))
    Bias.append(round((out1[n,:] - predy[n,:]).mean(),3))
  AssRMS = []
  AssBias = []
  for n in range(len(assimilation_out)):
    AssRMS.append(np.sqrt(((np.array(assimilation_out)[n,:] - predy[n,:]) ** 2).mean()))
    AssBias.append(round((np.array(assimilation_out)[n,:] - predy[n,:]).mean(),3))
  ALL_pred.append(out1)

plt.figure(figsize=(8,6))

plt.plot(RMSE[:-1],label="RMS error Machine Learning model + OI assimilation data",color = "blue")
plt.plot(Bias,"--",label="Bias Machine Learning model + OI assimilation data",color = "blue")
plt.plot(AssRMS,label="RMS error assimilation data",color = "red")
plt.plot(AssBias,"--",label="Bias assimilation data",color = "red")
plt.xlabel("times", size=18)
plt.ylabel("RMS errors", size=18)
plt.title('Compare with origional assimilation RMS error', size=20)
plt.legend(loc='upper right', numpoints=1, prop={'size':10})
plt.show()