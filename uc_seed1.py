#Gene wise lambda calculation
##python /home/dokada/Dropbox/analysis/2021.7/uc_seed1_0518.py
import pandas as pd
import numpy as np
import  os
import shutil
from multiprocessing import Pool
import glob
import matplotlib.pyplot as plt
from statsmodels.multivariate.manova import MANOVA
import scipy.stats as st

#file path
out_path = "/home/dokada/work_dir/uc_seed1/"
if os.path.isdir(out_path):
    shutil.rmtree(out_path)
    os.mkdir(out_path)
else:
    os.mkdir(out_path)
data_path = "/home/dokada/work_dir/pbmc_prepro0518/prepro/"
files = np.sort(glob.glob(data_path + "*"))
nn = len(files)
p = pd.read_csv(files[0],header=0).shape[1]

#True label
true_labels = pd.read_csv("/home/dokada/work_dir/pbmc_prepro0518/true_lab.csv",index_col=0).values[:,0]

#Set parameter
iter = p
ncores = 1
n_pic =  min([pd.read_csv(files[i],header=0).shape[0] for i in range(nn)])

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

#Calculation
np.random.seed(seed=1000)
pval_top = np.zeros(p)
sep_scores_top = np.zeros(p)
pval_top_rand = np.zeros(p)
sep_scores_top_rand = np.zeros(p)
for k in range(p):
    #Resampling
    #resampled_files = list()
    cellmararray1 = np.zeros([n_pic, 1, nn])
    np.random.seed(seed=k)
    for i in range(nn):
        file_path = files[i]
        with open(file_path,'rb') as f:
            nm = sum(1 for line in f)
        nm = nm - 1
        picked_idx = np.random.choice(nm, n_pic, replace=False)
        skipped_idx =  np.array([i for i in range(nm) if i not in picked_idx]) + 1
        expr_sub = pd.read_csv(file_path,skiprows=skipped_idx)
        if i == 0:
            gene_names = expr_sub.columns.values
        expr_sub = expr_sub.values
        cellmararray1[:,0,i] = expr_sub[:,k]
        print(i, "\n")
    print("Gene",k,"Resampling Finished","\n")

    #IP calculation
    def wrapper_kernel_ip(args):
        return kernel_ip(args, cellmararray=cellmararray1)

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
    print("Gene",k,"IP calculation Done.","\n")

    #DEEF
    Sita,eigen_value = df_simu_theo3_ip(ip_mat2)
    Sita_posi = np.real(Sita[:,eigen_value>=0])
    eig_posi = np.real(eigen_value[eigen_value>=0])

    #Calculate Separation Scores
    X = Sita_posi
    y = np.where(true_labels=="C", 0, 1)

    #Separation score with top 2 theta
    idx = 2
    X_sub = X[:,0:idx]
    df_data = pd.DataFrame(X_sub)
    m = df_data.shape[1]
    df_data.columns = ["theta" + str(i+1) for i in range(m)]
    lab = pd.DataFrame(y)
    lab.columns = ["label"]
    df = pd.concat([df_data, lab], axis=1)
    command = df_data.columns[0]
    for k1 in range(1, m):
        command = command + " + " + df_data.columns[k1]
    command = command + " ~ label"
    manova_obj = MANOVA.from_formula(command, data=df)
    pval_top[k] = manova_obj.mv_test().results['label']["stat"]["Pr > F"]["Wilks' lambda"]
    sep_scores_top[k] = manova_obj.mv_test().results['label']["stat"]["Value"]["Wilks' lambda"]
    print("Gene",k,"Pvalue.",pval_top[k],"\n")

    #Permutation
    y_rand = y.copy()
    np.random.shuffle(y_rand)
    lab = pd.DataFrame(y_rand)
    lab.columns = ["label"]
    df = pd.concat([df_data, lab], axis=1)
    command = df_data.columns[0]
    for k1 in range(1, m):
        command = command + " + " + df_data.columns[k1]
    command = command + " ~ label"
    manova_obj = MANOVA.from_formula(command, data=df)
    pval_top_rand[k] = manova_obj.mv_test().results['label']["stat"]["Pr > F"]["Wilks' lambda"]
    sep_scores_top_rand[k] = manova_obj.mv_test().results['label']["stat"]["Value"]["Wilks' lambda"]
    print("Random",k,"Pvalue.",pval_top_rand[k],"\n")


#histogram of score
fig = plt.figure()
plt.hist(pval_top)
fig.savefig(out_path + "pval.png")
plt.close()

#histogram of score
fig = plt.figure()
plt.hist(pval_top_rand)
fig.savefig(out_path + "pval_rand.png")
plt.close()

#histogram of score
fig = plt.figure()
plt.hist(sep_scores_top)
fig.savefig(out_path + "sep_scores.png")
plt.close()

#histogram of score
fig = plt.figure()
plt.hist(sep_scores_top_rand)
fig.savefig(out_path + "sep_scores_rand.png")
plt.close()

#Output
d_out ={"Gene_Name":gene_names,
        "lambda":sep_scores_top,
        "Null_lambda":sep_scores_top_rand,
        "P_value":pval_top,
        "NULL_P":pval_top_rand}
df_out = pd.DataFrame(d_out)
df_out.to_csv(out_path + "clf_scores_res.csv")

##QQ plot
#iters = len(pval)
observed = np.sort(pval_top)
#expected = st.uniform.ppf((np.arange(iters)+1)/(iters + 1))
expected = np.sort(pval_top_rand)
log_exp = -np.log10(expected)
log_obs = -np.log10(observed)
max_val = max(log_exp)
fig = plt.figure()
plt.plot([0,max_val],[0,max_val])
plt.plot(log_exp, log_obs,'r*')
plt.xlabel('Extected', fontsize=18)
plt.ylabel('Observed', fontsize=18)
fig.savefig(out_path + "pval_qq.png")
plt.close()

##QQ plot of lambda
#iters = len(pval)
observed = np.sort(sep_scores_top)
expected = np.sort(sep_scores_top_rand)
log_exp = -np.log10(expected)
log_obs = -np.log10(observed)
max_val = max(log_exp) + 0.3
fig = plt.figure()
plt.plot([0,max_val],[0,max_val])
plt.plot(log_exp, log_obs,'r*')
plt.xlabel('Extected', fontsize=18)
plt.ylabel('Observed', fontsize=18)
fig.savefig(out_path + "score_qq.png")
plt.close()
