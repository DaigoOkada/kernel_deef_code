#HVTN
import pandas as pd
import numpy as np
import  os
import shutil
from multiprocessing import Pool
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from statistics import mode
import xgboost as xgb
import glob
import matplotlib.pyplot as plt
import seaborn
from operator import itemgetter

#ファイルパスの設定
out_path = "/home/dokada/work_dir/hvtn_kdeef/"
if os.path.isdir(out_path):
    shutil.rmtree(out_path)
    os.mkdir(out_path)
else:
    os.mkdir(out_path)
data_path = "/home/dokada/work_dir/hvtn_prepro0518/trans_data/"
annot = pd.read_csv("/home/dokada/work_dir/hvtn_prepro0518/annot_stm.csv")
files = data_path + annot["FCS.File"].values + ".csv"
nn = len(files)

#Define Trainn data and Test data
lab1 = 'ENV-1-PTEG'
lab2 = 'GAG-1-PTEG'
train_idx = np.where(annot["Sample.Characteristic"].values=="training")[0]
test_idx = np.where(annot["Sample.Characteristic"].values=="testing")[0]
y_labels = annot["Sample.Treatment"].values
y_labels_bin = np.where(np.array(y_labels)==lab1, 0, 1)

#Set parameter
iter = 25
ncores = 45
flg1_thres = 10
cv_num = 5
n_pic = 1000
n_estimators = 5000
p = pd.read_csv(files[0],header=0).shape[1]

#ここからはデータセットによらずに共通
#結果を格納するオブジェクト
prdct_mat = np.zeros([iter,len(test_idx)], dtype="object")
prdct_mat_prob = np.zeros([iter,len(test_idx)], dtype="object")
cv_iter_testacc =  np.zeros(iter)
cv_iter_sigjiku = np.zeros(iter)
cv_iter_train_bestacc =  np.zeros(iter)
cv_iter_maxf = np.zeros(iter)
cv_iter_maxd = np.zeros(iter)

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

    #sig_jikuの組み合わせでCV
    X_train1 = Sita_posi[train_idx,:]
    y_train1_bin = y_labels_bin[train_idx]

    #Grid search
    max_posi_num = Sita_posi.shape[1]
    sig_can = np.array(range(1, max_posi_num+1)) #1 ~ max_posi_numまで生成
    sig_bestacc_vec = list()
    best_acc = 0
    best_params = None
    best_est = None
    flg1 = 0
    sig_jiku_vec = list()
    for sig_j in range(len(sig_can)):
        sig_jiku = int(sig_can[sig_j])
        X_train1_sub = X_train1[:,0:sig_jiku].reshape([X_train1.shape[0], sig_jiku])
        def param():
            ret = {
                    'max_depth':[2, 3, 4, 5]
            }
            return ret
        clf = xgb.XGBClassifier(random_state=k, n_estimators=n_estimators)
        cv_obj = GridSearchCV(clf, param(), cv=cv_num, verbose=0, n_jobs=ncores)
        cv_obj.fit(X_train1_sub, y_train1_bin)
        tmp_best_param = cv_obj.best_params_
        tmp_acc = cv_obj.best_score_
        sig_bestacc_vec.append(tmp_acc)
        sig_jiku_vec.append(sig_jiku)
        print("sig_j",sig_j,"tmp_acc:",tmp_acc,"best_acc:",best_acc,":Coordinate searching.","\n")
        if tmp_acc > best_acc:
            best_acc = tmp_acc
            best_sigjiku = sig_jiku
            best_param = tmp_best_param
            best_est = cv_obj.best_estimator_
            flg1 = 0
            print("sig_j",sig_j,":flg1_Reset.","\n")
        else:
            flg1 = flg1 + 1
            print("sig_j",sig_j,":flg1_Update.","\n")
        if flg1 > flg1_thres:
            break
    cv_iter_train_bestacc[k] = best_acc
    cv_iter_sigjiku[k] = best_sigjiku
    cv_iter_maxd[k] = best_param["max_depth"]
    print("Iteration",k,"Coordinate search Done.","\n")

    #sig_jikuとtrain_accの関係
    fig = plt.figure()
    plt.plot(sig_jiku_vec, sig_bestacc_vec)
    plt.ylim([0,1])
    plt.xlabel("Num of coordinates", fontsize=18)
    plt.ylabel("Best CV scores", fontsize=18)
    plt.savefig(out_path + "train_accplot" + str(k) + ".png")
    plt.close()

    #Create model with best paramete
    X_train1 =  Sita_posi[train_idx,0:(best_sigjiku)]
    y_train1_bin =  y_labels[train_idx]
    feature_names = ["theta" + str(i + 1) for i in range(X_train1.shape[1])]

    #Importanceのplot
    bst = best_est.get_booster()
    bst.feature_names = feature_names
    xgb.plot_importance(bst)
    plt.subplots_adjust(left=0.2)
    plt.savefig(out_path + "importance" + str(k) + ".png")
    plt.close()

    #importanceが上位のtheta座標のプロット
    mat_fscore = pd.DataFrame(bst.get_fscore().items()).values
    coordinate_num = mat_fscore.shape[0]
    if coordinate_num > 1:
        att = [[mat_fscore[i,0], -int(mat_fscore[i,1]), int(mat_fscore[i,0].split("theta")[1])] for i in range(coordinate_num)]
        att_sorted = sorted(att, key=itemgetter(1,2), reverse=False)
        top_coord1 = att_sorted[0][0]
        top_coord2 = att_sorted[1][0]
        target_theta_idx1 = np.where(np.array(feature_names)==np.array(top_coord1))[0][0]
        target_theta_idx2 = np.where(np.array(feature_names)==np.array(top_coord2))[0][0]

        #plot of important theta
        fig = plt.figure()
        colors = np.ones(len(y_labels),dtype="object")
        colors[y_labels == lab1] =  "red"
        colors[y_labels == lab2] =  "blue"
        xlab = feature_names[target_theta_idx1]
        ylab = feature_names[target_theta_idx2]
        plt.scatter(Sita_posi[:,target_theta_idx1], Sita_posi[:,target_theta_idx2], color=colors)
        plt.title("Important coordinates", fontsize=18)
        plt.subplots_adjust(left=0.2)
        plt.xlabel(xlab, fontsize=18)
        plt.ylabel(ylab, fontsize=18)
        fig.savefig(out_path + "imptheta_iter" + str(k) + ".png")
        plt.close()
    else:
        target_theta_idx1 = 0
        target_theta_idx2 = 1
        #plot of important theta
        fig = plt.figure()
        colors = np.ones(len(y_labels),dtype="object")
        colors[y_labels == lab1] =  "red"
        colors[y_labels == lab2] =  "blue"
        xlab = "theta1"
        ylab = "theta2"
        plt.scatter(Sita_posi[:,target_theta_idx1], Sita_posi[:,target_theta_idx2], color=colors)
        plt.title("Important coordinates", fontsize=18)
        plt.subplots_adjust(left=0.2)
        plt.xlabel(xlab, fontsize=18)
        plt.ylabel(ylab, fontsize=18)
        fig.savefig(out_path + "imptheta_iter" + str(k) + ".png")
        plt.close()

    #plot of theta1 and theta2
    fig = plt.figure()
    colors = np.ones(len(y_labels),dtype="object")
    colors[y_labels == lab1] =  "red"
    colors[y_labels == lab2] =  "blue"
    plt.scatter(Sita_posi[:,0],Sita_posi[:,1],color=colors)
    plt.title("Top coordinates", fontsize=18)
    plt.subplots_adjust(left=0.2)
    plt.xlabel("theta1", fontsize=18)
    plt.ylabel("theta2", fontsize=18)
    fig.savefig(out_path + "toptheta_iter" + str(k) + ".png")
    plt.close()

    #Predict
    X_test1 =  Sita_posi[test_idx,0:(best_sigjiku)]
    test_xgb = xgb.DMatrix(X_test1,feature_names=feature_names)
    prdct_prob = bst.predict(test_xgb)
    y_pred = np.where(prdct_prob < 0.5, 0, 1)
    prdct_mat[k,:] = y_pred
    prdct_mat_prob[k,:] = prdct_prob

    #Test
    y_test1_bin =  y_labels_bin[test_idx]
    cm = confusion_matrix(y_pred, y_test1_bin)
    k_acc = np.diag(cm).sum()/cm.sum()
    cv_iter_testacc[k] = k_acc
    print("Iteration",k,"Acc: ", k_acc, "\n")

#重みパラメータも出力
d ={"cv_iter_sigjiku":cv_iter_sigjiku,
    "cv_iter_testacc":cv_iter_testacc,
    "cv_iter_train_bestacc":cv_iter_train_bestacc,
    "cv_iter_max_depth":cv_iter_maxd}
df = pd.DataFrame(d)
df.to_csv(out_path+"cv_weight_paras.csv")

#多数決を取る
df = pd.DataFrame(prdct_mat)
df.to_csv(out_path+"prect_mat.csv")
a = iter/2
y_test1_bin =  y_labels_bin[test_idx]
prdct_con = [1 if sum(prdct_mat[:,i]==1) > a else 0 for i in range(prdct_mat.shape[1])]
cm_con = confusion_matrix(prdct_con, y_test1_bin)
k_acc = np.diag(cm_con).sum()/cm_con.sum()
df = pd.DataFrame(cm_con)
df.to_csv(out_path+"cm_con.csv")

#AccのViolin plot
fig = plt.figure()
df = pd.DataFrame(cv_iter_testacc, columns=["Resampling"])
seaborn.violinplot(data=df, color="0.85")
seaborn.stripplot(data=df)
plt.ylim([-0.1,1.1])
plt.axhline(y=k_acc, color="red")
fig.savefig(out_path + "acctest_violin" + str(k) + ".png")

#ACCが最大の反復(2021.4.6追記)
with open('max_acc_note.txt', 'w') as f:
     print("max_test_acc_iter : iter" + str(np.argsort(-cv_iter_testacc)[0]),file=f)
     print("k_acc : " + str(k_acc),file=f)
