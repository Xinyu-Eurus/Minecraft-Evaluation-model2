#test_RF-2.py
from train_RF2 import FEATURES_NAME, FEATURES_NAME_0to5, prepare_Xy

import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import dump, load

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def k_fold(X_train, y_train, k=4, n=10, max_depth=10, n_estimators=10, min_samples_leaf=2, max_features="log2", store_model=False, print_res=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=204)
    clf = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=1, n_estimators=n_estimators)
    
    accs=[]
    train_accs=[]
    preds=[]
    gts=[]
    conf_mats=[]
    classes=[1,2,3]
    for i in range(n):
#         print("No : {}".format(i))
        for train, dev in kf.split(X_train, y_train):
            clf.fit(X_train[train], y_train[train])
            preds.append(clf.predict(X_train[dev]).tolist())
            gts.append(y_train[dev].tolist())
            conf_mat = confusion_matrix(y_train[dev], clf.predict(X_train[dev]))
            conf_mats.append(conf_mat)
            
            acc = clf.score(X_train[dev], y_train[dev])
            train_acc=clf.score(X_train[train], y_train[train])
            accs.append(acc)
            train_accs.append(train_acc)
#             print("gts:", gts, "preds:",preds)
#             print("acc : {}".format(acc))
#             print("feature_importances : {}".format(clf.feature_importances_)) 
            
    accs=np.array(accs)
    acc_ave=np.mean(accs)
    acc_std=np.std(accs)

    train_accs=np.array(train_accs)
    train_acc_ave=np.mean(train_accs)
    train_acc_std=np.std(train_accs)
    
    gts_array=np.array(sum(gts,[])) 
    preds_array=np.array(sum(preds,[]))
   
    confusion_mat = confusion_matrix(gts_array, preds_array)

    clf.fit(X_train, y_train)
    final_Acc=clf.score(X_train, y_train)
    
    if print_res:
        print("\ntest_acc_ave : {}, test_acc_std:{}".format(acc_ave, acc_std))
        print("\ntrain_acc_ave : {}, train_acc_std:{}".format(train_acc_ave, train_acc_std))
        print("final_Acc:", final_Acc)

        print(gts_array.shape, preds_array.shape)
        print("gts:", gts_array, "preds:",preds_array)
        print("confusion_mat : {}".format(confusion_mat))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
        disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal", values_format="d")
        plt.show()

    if store_model:
        dump(clf, './models/RF-2.model')
    
    return acc_ave, acc_std, clf, train_acc_ave, train_acc_std, final_Acc, confusion_mat
        

def final_test(clf, X_test, y_test):
    classes=[1,2,3]
    acc = clf.score(X_test, y_test)
    print("acc : {}".format(acc))
    print("feature_importances : {}".format(clf.feature_importances_)) 
    
    confusion_mat = confusion_matrix(y_test, clf.predict(X_test))
    print("confusion_mat : {}".format(confusion_mat))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal", values_format="d")
    plt.show()
    return acc

def repeat_test(df, n=10, max_depth=5, n_estimators=10, min_samples_leaf=4):
    X_train, X_test, y_train, y_test=prepare_Xy(df)
    X=np.concatenate((X_train, X_test))
    y=np.concatenate((y_train, y_test))
    acc_aves=[]
    acc_stds=[]
    train_acc_aves=[]
    train_acc_stds=[]
    final_Accs=[]
    confusion_mats=[]
    
    for i in range(n): 
        if i%10==9:
            print("No : {}".format(i))
        acc_ave, acc_std, clf, train_acc_ave, train_acc_std, final_Acc, confusion_mat=k_fold(X, y, k=10, n=1, 
                             max_depth=max_depth, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf) 
        acc_aves.append(acc_ave)
        acc_stds.append(acc_std)
        train_acc_aves.append(train_acc_ave)
        train_acc_stds.append(train_acc_std)
        final_Accs.append(final_Acc)
        confusion_mats.append(confusion_mat)
        
        X, y = sklearn.utils.shuffle(X, y) 
    acc_aves=np.array(acc_aves)
    acc_stds=np.array(acc_stds)
    train_acc_aves=np.array(train_acc_aves)
    train_acc_stds=np.array(train_acc_stds)
    final_Accs=np.array(final_Accs)
    confusion_mats=np.array(confusion_mats)
    print("\nacc_aves_ave : {:.4}, acc_aves_std : {:.4}, acc_stds_ave:{:.4}"
          .format(np.mean(acc_aves),np.std(acc_aves), np.mean(acc_stds)))
    print("train_acc_aves_ave : {:.4}, train_acc_aves_std : {:.4}, train_acc_stds_ave:{:.4}"
          .format(np.mean(train_acc_aves),np.std(train_acc_aves), np.mean(train_acc_stds)))
    print("final_Accs_ave : {:.4}, final_Accs_std : {:.4}"
          .format(np.mean(final_Accs),np.std(final_Accs)))
    confusion_mats_sum=np.sum(confusion_mats,axis=0)
    print(confusion_mats_sum)
    classes=[1,2,3]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mats_sum, display_labels=classes)
    disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal", values_format="d")
    plt.show()
    
    return confusion_mats


if __name__ == "__main__":
    FILEPATH = sys.argv[1]
    df = pd.read_csv(FILEPATH+"/fea_m_compare.csv") 
    # df = pd.read_csv(FILEPATH+"/fea_m_all.csv") 
    df.attack_effi[df.attack_effi>0.05] = 0.05

    confusion_mats=repeat_test(df, n=50, max_depth=5, n_estimators=50, min_samples_leaf=4)