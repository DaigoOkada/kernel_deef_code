#HEU data
import pandas as pd
import numpy as np
import  os
import shutil
from scipy.spatial import distance_matrix
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from statistics import mode
import xgboost as xgb
import glob
import matplotlib.pyplot as plt
import seaborn
from operator import itemgetter


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

#ファイルパスの設定
annot =pd.read_csv("/home/dokada/work_dir/heu_prepro0518/annot.csv",index_col=0)
out_path = "/home/dokada/work_dir/heu_kdeef/"
if os.path.isdir(out_path):
    shutil.rmtree(out_path)
    os.mkdir(out_path)
else:
    os.mkdir(out_path)
data_path = "/home/dokada/work_dir/heu_prepro0518/trans_data/"

#Define Train data and Test data
train_pro = 0.5
ind_num = 44
tmp = train_test_split(np.arange(ind_num),train_size=train_pro, random_state=1000)
train_ind_idx = np.sort(tmp[0]) #この順番を使う
test_ind_idx = np.sort(tmp[1]) #この順番を使う

#Set parameter
iters = 25
ncores = 45
flg1_thres = 10
cv_num = 5
n_pic = 1000
lab1 = 'HEU'
lab2 = 'UE'
eig_alpha = 0.8
n_estimators = 5000


#特定の処理群　+ unstimのみ抽出
doses = ['CPG', 'LPS', 'PAM', 'PG', 'PIC', 'R848','unstim']
for dos in doses:
    #dos ="CPG"
    annot_dos = annot[annot["dose"].values==dos]
    files = data_path + annot_dos["FCS.file"].values + ".csv"
    nn = len(files)
    p = pd.read_csv(files[0],header=0).shape[1]
    annot_dos_train = annot_dos.values[train_ind_idx,:]
    annot_dos_test = annot_dos.values[test_ind_idx,:]

    #True labels
    y_labels =  annot_dos["Condition"].values
    y_train1_true = y_labels[train_ind_idx]
    y_test1_true = y_labels[test_ind_idx]

    #binary label
    y_labels_bin = np.where(np.array(y_labels)==lab1, 0, 1)
    y_train1_true_bin = np.where(np.array(y_train1_true)==lab1, 0, 1)
    y_test1_true_bin = np.where(np.array(y_test1_true)==lab1, 0, 1)


    #結果を格納するオブジェクト
    prdct_mat = np.zeros([iters,len(test_ind_idx)])
    prdct_mat_prob = np.zeros([iters,len(test_ind_idx)])
    cv_iter_testacc =  np.zeros(iters)
    cv_iter_sigjiku = np.zeros(iters)
    cv_iter_train_bestacc =  np.zeros(iters)

    #計算
    for k in range(iters):
        #Resampling
        #resampled_files = list()
        cellmararray1 = np.zeros([n_pic, p, nn])
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
            cellmararray1[:,:,i] = expr_sub
            print(i, "\n")
        print("Dos", dos, "Iteration",k,"Resampling Finished","\n")

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
        print("Dos", dos, "Iteration",k,"IP calculation Done.","\n")

        #DEEF
        Sita, eigen_value = df_simu_theo3_ip(ip_mat2)
        Sita_posi = Sita[:,eigen_value>=0]
        eig_posi = eigen_value[eigen_value>=0]
        X_train1_full = Sita_posi[train_ind_idx,:]
        X_test1_full = Sita_posi[test_ind_idx,:]

        #Output
        df = pd.DataFrame(X_train1_full)
        df.to_csv(out_path + dos + "_X_train1_full_" + "iter" + str(k) + ".csv")
        df = pd.DataFrame(X_test1_full)
        df.to_csv(out_path + dos + "_X_test1_full_" + "iter" + str(k) + ".csv")
        df = pd.DataFrame(eig_posi)
        df.to_csv(out_path + dos + "_eigposi_" + "iter" + str(k) + ".csv")


        #ML for each dose
        max_posi_num = X_train1_full.shape[1]
        sig_can = np.array(range(1,max_posi_num+1))
        sig_jiku_vec = list()
        sig_bestacc_vec = list()
        best_acc = 0
        best_params = None
        best_est = None
        best_train_data = None
        best_sigjiku = None
        flg1 = 0
        doses_two = [dos] #ここが違う
        for sig_j in range(len(sig_can)):
            sig_jiku = int(sig_can[sig_j])
            X_train1 = X_train1_full[:,0:sig_jiku]
            X_test1 = X_test1_full[:,0:sig_jiku]

            #ML
            clf = xgb.XGBClassifier(random_state=k, n_estimators=n_estimators)
            def param():
                ret = {
                        'max_depth':[2, 3, 4, 5]
                }
                return ret
            cv_obj = GridSearchCV(clf, param(), cv=cv_num, verbose=0, n_jobs=ncores)
            cv_obj.fit(X_train1, y_train1_true_bin)
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
                best_train_data = X_train1
                best_test_data = X_test1
                flg1 = 0
                print("sig_j",sig_j,":flg1_Reset.","\n")
            else:
                flg1 = flg1 + 1
                print("sig_j",sig_j,":flg1_Update.","\n")
            if flg1 > flg1_thres:
                break

        #Create model
        cv_iter_train_bestacc[k] = best_acc
        cv_iter_sigjiku[k] = best_sigjiku
        print("Iteration",k,"Coordinate search Done.","\n")

        #sig_jikuとtrain_accの関係のplot
        fig = plt.figure()
        plt.plot(sig_jiku_vec, sig_bestacc_vec)
        plt.ylim([0,1])
        plt.xlabel("Num of coordinates", fontsize=18)
        plt.ylabel("Best CV scores", fontsize=18)
        plt.savefig(out_path + dos + "_train_accplot" + str(k) + ".png")
        plt.close()

        #With title version
        fig = plt.figure()
        plt.plot(sig_jiku_vec, sig_bestacc_vec)
        plt.ylim([0,1])
        plt.xlabel("Num of coordinates", fontsize=18)
        plt.ylabel("Best CV scores", fontsize=18)
        plt.title("Resampling" + str(k + 1), fontsize=18)
        plt.savefig(out_path + dos + "_train_accplot_title" + str(k) + ".png")
        plt.close()

        #Create model with best paramete
        X_train1 =  best_train_data
        X_test1 =  best_test_data
        feature_names = [d_name + "_theta" + str(i + 1) for d_name in doses_two for i in range(best_sigjiku)]


        #Importanceのplot
        bst = best_est.get_booster()
        bst.feature_names = feature_names
        xgb.plot_importance(bst)
        plt.subplots_adjust(left=0.3)
        plt.savefig(out_path + dos + "_importance" + str(k) + ".png")
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
            xlab = feature_names[target_theta_idx1]
            ylab = feature_names[target_theta_idx2]
        else:
            target_theta_idx1 = 0
            target_theta_idx2 = 1
            xlab = "theta1"
            ylab = "theta2"
        #plot of important theta
        fig = plt.figure()
        colors = np.ones(len(y_labels),dtype="object")
        colors[y_labels == lab1] =  "red"
        colors[y_labels == lab2] =  "blue"
        plt.scatter(Sita_posi[:,target_theta_idx1], Sita_posi[:,target_theta_idx2],color=colors)
        plt.title("Important coordinates", fontsize=18)
        plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.xlabel(xlab, fontsize=18)
        plt.ylabel(ylab, fontsize=18)
        fig.savefig(out_path + str(dos) + "_imptheta_iter" + str(k) + ".png")
        plt.close()

        #Output
        df = pd.DataFrame(X_train1)
        df.to_csv(out_path + dos + "_Xtrain1_" + "iter" + str(k) + ".csv")

        df = pd.DataFrame(X_test1)
        df.to_csv(out_path + dos + "_Xtest1_" + "iter" + str(k) + ".csv")

        df = pd.DataFrame(y_train1_true_bin)
        df.to_csv(out_path + dos + "_ytrain1_" + "iter" + str(k) + ".csv")

        df = pd.DataFrame(y_test1_true_bin)
        df.to_csv(out_path + dos + "_ytest1_" + "iter" + str(k) + ".csv")


        #Create Test
        test_xgb = xgb.DMatrix(X_test1, feature_names=feature_names)
        prdct_prob = bst.predict(test_xgb)
        y_pred = np.where(prdct_prob < 0.5, 0, 1)
        prdct_mat[k,:] = y_pred
        prdct_mat_prob[k,:] = prdct_prob

        #Test
        cm = confusion_matrix(y_pred, y_test1_true_bin)
        k_acc = np.diag(cm).sum()/cm.sum()
        cv_iter_testacc[k] = k_acc
        print("Dos", dos, "Iteration", k, "Acc",str(k_acc),"\n")


    #多数決を取る
    df = pd.DataFrame(prdct_mat)
    df.to_csv(out_path + dos + "_prect_mat.csv")
    a = iters/2
    prdct_con = np.where(np.sum(prdct_mat,axis=0) < a, 0, 1)
    cm_con = confusion_matrix(prdct_con, y_test1_true_bin)
    k_acc = np.diag(cm_con).sum()/cm_con.sum()
    df = pd.DataFrame(cm_con)
    df.to_csv(out_path + dos + "_cm_con.csv")

    #重みパラメータも出力
    d ={"cv_iter_sigjiku":cv_iter_sigjiku,
        "cv_iter_testacc":cv_iter_testacc,
        "cv_iter_train_bestacc":cv_iter_train_bestacc}
    df = pd.DataFrame(d)
    df.to_csv(out_path + dos + "_cv_weight_paras.csv")

    #AccのViolin plot
    fig = plt.figure()
    df = pd.DataFrame(cv_iter_testacc, columns=["Resampling"])
    seaborn.violinplot(data=df, color="0.85")
    seaborn.stripplot(data=df)
    plt.ylim([-0.1,1.1])
    plt.title(dos, fontsize=18)
    plt.axhline(y=k_acc, color="red")
    fig.savefig(out_path + dos + "_acctest_violin.png")
