#kernelDEEF for UC(2021.6.20)
#python /home/dokada/Dropbox/analysis/2021/4/kdeef_uc.py
#Import modules
import pandas as pd
import numpy as np
import  os
import shutil
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt

#User Set parameter
out_path = "/home/dokada/work_dir/kdeef_uc/"
if os.path.isdir(out_path):
    shutil.rmtree(out_path)
    os.mkdir(out_path)
else:
    os.mkdir(out_path)
ncores = 3 #number of cores of parallel computing

#Load Data
dat = np.load("/home/dokada/work_dir/scrnaseq_dataset/uc_data.npy") #cell * gene * subject array
y_labels = pd.read_csv("/home/dokada/work_dir/pbmc_prepro0518/true_lab.csv",index_col=0).values[:,0] #labels of subjects
fn = len(y_labels) # number of subjects

#Define the function for paralell computing
def kernel_ip(sample_idx, cellmararray):
    data1 = cellmararray[:,:,sample_idx]
    n1 = cellmararray.shape[0]
    p = cellmararray.shape[1]
    nn = cellmararray.shape[2]
    gamma = 1/p
    ip_row = np.zeros(nn)
    for j in range(sample_idx, nn):
        data2 = cellmararray[:,:,j]
        n2 = data2.shape[0]
        ip_vec = np.zeros(n1)
        for i in range(n1):
            ip_vec[i] = np.exp((((data2 - data1[i,])**2).sum(axis=1))*(-gamma)).sum()
        ip_row[j] = ip_vec.sum()/(n1*n2)
    print(sample_idx,"\n")
    return(ip_row)

def wrapper_kernel_ip(args):
    return kernel_ip(args, cellmararray=dat)

#calculation
with Pool(processes=ncores) as pro:
        res = pro.map(wrapper_kernel_ip,range(fn))

ip_mat = np.zeros([fn,fn])
for i in range(len(res)):
    ip_mat[i,:] = res[i]
ip_mat2 = np.zeros([fn, fn])
for i in range(fn):
    for j in range(i,fn):
        ip_mat2[i,j] = ip_mat[i,j]
        ip_mat2[j,i] = ip_mat[i,j]

#Decomposition of inner product matrix
def deef(ip_mat):
    ip_mat = np.asarray(ip_mat)
    sita_ip_est_mat = np.log(ip_mat)/2
    eigen_value, V = np.linalg.eig(sita_ip_est_mat)
    Sigma = np.diag(np.sqrt(np.abs(eigen_value)))
    Sita = np.dot(V,Sigma)
    idx = np.argsort(-eigen_value)
    eigen_value = eigen_value[idx]
    Sita = Sita[:,idx]
    res = [Sita,eigen_value]
    return(res)

Theta_all, eigen_value = deef(ip_mat2)
Theta = Theta_all[:,eigen_value>=0]

#Visualization
fig = plt.figure()
colors = np.ones(fn, dtype="object")
colors[y_labels == "C"] =  "red"
colors[y_labels == "U"] =  "blue"
plt.scatter(Theta[:, 0], Theta[:, 1],color=colors)
plt.title("Theta coordinates", fontsize=18)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.xlabel("theta1", fontsize=18)
plt.ylabel("theta2", fontsize=18)
fig.savefig(out_path + "theta_plot.png")
