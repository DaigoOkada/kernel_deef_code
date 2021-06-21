#integrated ML for Heu data
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

#Out path
out_path = "/home/dokada/work_dir/heu_ML/"
if os.path.isdir(out_path):
    shutil.rmtree(out_path)
    os.mkdir(out_path)
else:
    os.mkdir(out_path)

#Setting
iters = 25
ncores = 45
flg1_thres = 10
cv_num = 5
n_pic = 1000
lab1 = 'HEU'
lab2 = 'UE'

doses = ['CPG', 'LPS', 'PAM', 'PG', 'PIC', 'R848', "unstim"]
data_path = "/home/dokada/work_dir/heu_kdeef/"
prdct_mat = np.zeros([iters, 22])
prdct_mat_prob = np.zeros([iters, 22])
cv_iter_testacc = np.zeros(iters)
for k in range(iters):
    feature_names = list()
    for d in range(len(doses)):
        dos = doses[d]
        Xtrain_file = dos + "_Xtrain1_iter" + str(k) + ".csv"
        ytrain_file = dos + "_ytrain1_iter" + str(k) + ".csv"
        Xtest_file = dos + "_Xtest1_iter" + str(k) + ".csv"
        ytest_file = dos + "_ytest1_iter" + str(k) + ".csv"
        X_train1_dose = pd.read_csv(data_path + Xtrain_file, index_col=0).values
        y_train1_dose = pd.read_csv(data_path + ytrain_file, index_col=0).values[:,0]
        X_test1_dose = pd.read_csv(data_path + Xtest_file, index_col=0).values
        y_test1_dose = pd.read_csv(data_path + ytest_file, index_col=0).values[:,0]
        theta_num = X_train1_dose.shape[1]
        feature_names_dose = [dos + "_theta" + str(i + 1) for i in range(theta_num)]
        feature_names = np.hstack([feature_names, feature_names_dose])
        if d == 0:
            X_train1 = X_train1_dose
            X_test1 = X_test1_dose
            y_train1_bin = y_train1_dose
            y_test1_bin = y_test1_dose
        else:
            X_train1 = np.hstack([X_train1, X_train1_dose])
            X_test1 = np.hstack([X_test1, X_test1_dose])
            if np.all(y_train1_bin == y_train1_dose) == False:
                raise ValueError("train data error")
            if np.all(y_test1_bin == y_test1_dose) == False:
                raise ValueError("test data error")

    #ML
    clf = xgb.XGBClassifier(random_state=k, n_estimators=5000)
    def param():
        ret = {
            'max_depth':[2, 3, 4, 5]
        }
        return ret
    cv_obj = GridSearchCV(clf, param(), cv=cv_num, verbose=0, n_jobs=ncores)
    cv_obj.fit(X_train1, y_train1_bin)
    best_est = cv_obj.best_estimator_
    bst = best_est.get_booster()
    #bst.feature_names = feature_names

    #Create Test
    #test_xgb = xgb.DMatrix(X_test1,feature_names=feature_names)
    test_xgb = xgb.DMatrix(X_test1)
    prdct_prob = bst.predict(test_xgb)
    y_pred = np.where(prdct_prob < 0.5, 0, 1)
    prdct_mat[k,:] = y_pred
    prdct_mat_prob[k,:] = prdct_prob

    #Test
    cm = confusion_matrix(y_pred, y_test1_bin)
    k_acc = np.diag(cm).sum()/cm.sum()
    cv_iter_testacc[k] = k_acc
    print("Iteration", k, "Acc",str(k_acc),"\n")


#多数決を取る
df = pd.DataFrame(prdct_mat)
df.to_csv(out_path + "prect_mat.csv")
df = pd.DataFrame(y_test1_bin)
df.to_csv(out_path + "y_teat1.csv")
a = iters/2
prdct_con = np.where(np.sum(prdct_mat,axis=0) < a, 0, 1)
cm_con = confusion_matrix(prdct_con, y_test1_bin)
k_acc = np.diag(cm_con).sum()/cm_con.sum()
df = pd.DataFrame(cm_con)
df.to_csv(out_path + "cm_con.csv")

#重みパラメータも出力
d ={"cv_iter_testacc":cv_iter_testacc}
df = pd.DataFrame(d)
df.to_csv(out_path + "cv_weight_paras.csv")

#AccのViolin plot
fig = plt.figure()
df = pd.DataFrame(cv_iter_testacc, columns=["Resampling"])
seaborn.violinplot(data=df, color="0.85")
seaborn.stripplot(data=df)
plt.title("Integrated", fontsize=18)
plt.ylim([-0.1,1.1])
plt.axhline(y=k_acc, color="red")
fig.savefig(out_path + "acctest_violin.png")
