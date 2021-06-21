#ITN (run:2021.6.20)
##python /home/dokada/Dropbox/analysis/2021.4/itn_kdeef.py
import pandas as pd
import numpy as np
import  os
import shutil
from multiprocessing import Pool
import glob
import matplotlib.pyplot as plt
import seaborn as sns

#ファイルパスの設定
annot = pd.read_csv("/home/dokada/work_dir/ITN_prepro0518/annot.csv", index_col=0)
out_path = "/home/dokada/work_dir/itn_kdeef/"
if os.path.isdir(out_path):
    shutil.rmtree(out_path)
    os.mkdir(out_path)
else:
    os.mkdir(out_path)
data_path = "/home/dokada/work_dir/ITN_prepro0518/trans_data/"
nn = len(glob.glob(data_path + "*"))
files =  np.array([data_path + str(i+1) + ".csv" for i in range(nn)])

#Define Trainn data and Test data
y_labels = annot["GroupID"].values

#Set parameter
iter = 3
ncores = 2
n_pic = 1000
p = pd.read_csv(files[0],header=0).shape[1]

#関数
def df_simu_theo3_ip(ip_mat):
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

#test ok
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

#計算
for k in range(iter):
    #Resampling
    cellmararray = np.zeros([n_pic, p, nn])
    np.random.seed(seed=k)
    for i in range(nn):
        file_path = files[i]
        with open(file_path,'rb') as f:
            nm = sum(1 for line in f)
        nm = nm - 1
        picked_idx = np.random.choice(nm, n_pic, replace=False) #重複あると次でエラーに
        skipped_idx =  np.array([i for i in range(nm) if i not in picked_idx]) + 1 #HEADERをたす
        expr_sub = pd.read_csv(file_path,skiprows=skipped_idx)
        expr_sub = expr_sub.values
        cellmararray[:,:,i] = expr_sub
        print(i, "\n")
    print("Iteration",k,"Resampling Finished","\n")

    #IP calculation
    def wrapper_kernel_ip(args):
        return kernel_ip(args, cellmararray=cellmararray)

    with Pool(processes=ncores) as pro:
            res = pro.map(wrapper_kernel_ip,range(nn))

    ip_mat = np.zeros([nn,nn])
    for i in range(len(res)):
        ip_mat[i,:] = res[i]
    ip_mat2 = np.zeros([nn,nn])
    for i in range(nn):
        for j in range(i,nn):
            ip_mat2[i,j] = ip_mat[i,j]
            ip_mat2[j,i] = ip_mat[i,j]
    print("Iteration",k,"IP calculation Done.","\n")

    #DEEF
    Sita,eigen_value = df_simu_theo3_ip(ip_mat2)
    Sita_posi = Sita[:,eigen_value>=0]
    eig_posi = eigen_value[eigen_value>=0]

    #Output
    df = pd.DataFrame(Sita_posi)
    df.to_csv(out_path+"X_" + "iter" + str(k) + ".csv")
    df = pd.DataFrame(ip_mat2)
    df.to_csv(out_path+"ipmat_" + "iter" + str(k) + ".csv")
    df = pd.DataFrame(eig_posi)
    df.to_csv(out_path+"eigposi_" + "iter" + str(k) + ".csv")


    #plot of theta1 and theta2
    fig = plt.figure()
    colors = np.ones(len(y_labels),dtype="object")
    colors[y_labels == "Group 1"] =  "red"
    colors[y_labels == 'Group 5'] =  "blue"
    colors[y_labels == 'Group 6'] =  "black"
    plt.scatter(Sita_posi[:,0], Sita_posi[:,1],color=colors)
    plt.title("Top coordinates", fontsize=18)
    plt.subplots_adjust(left=0.2)
    plt.xlabel("theta1", fontsize=18)
    plt.ylabel("theta2", fontsize=18)
    fig.savefig(out_path + "toptheta_iter" + str(k) + ".png")
    plt.close()

    #Pair plot
    sig_jiku = 3
    feature_names = ["theta" + str(i+1) for i in range(Sita_posi.shape[1])]
    df = pd.DataFrame(Sita_posi[:,0:sig_jiku],columns=feature_names[0:sig_jiku])
    df["label"] = y_labels
    fig = sns.pairplot(df,  hue="label")
    fig.savefig(out_path + "pairs_toptheta_iter" + str(k) + ".png")
    plt.close()
