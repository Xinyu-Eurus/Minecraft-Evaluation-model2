#train_RF-2.py
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


ITEMS_NAME=["COAL", "COBBLESTONE", "CRAFTING_TABLE", "DIRT", "FURNACE", "IRON_AXE", "IRON_INGOT", "IRON_ORE", "IRON_PICKAXE", \
          "LOG", "PLANKS", "STICK", "STONE", "STONE_AXE", "STONE_PICKAXE", "TORCH", "WOODEN_AXE", "WOODEN_PICKAXE"] 

FEATURES_NAME=["inventory_firstGainOrder", "inventory_firstGainStep",\
            "inventory_accum_reward", "inventory_rewardedGainStep", "inventory_rewardedGainOrder",\
                     "if_iron_axe", "if_stone_axe", "if_wooden_axe", "sparse_reward", "dense_reward", \
            "attack_effi", "attack_ratio", "attack_equipped", "camera_mov_ratio", "position_mov_ratio", \
            "torch_placed", "cobblestone_placed", "dirt_placed", "stone_placed", "if_smelt_coal"]

def generate_0to5_names():
    names = []
    for i in range(5):
        for j in range(18):
            names.append(FEATURES_NAME[i]+"_"+str(j))
    return names

FEATURES_NAME_0to5=generate_0to5_names()


def prepare_Xy(df, test_size = 0.3):
    X=[]
    for fea in FEATURES_NAME:
        X.append(df[fea])
    X.append(df["ai_eva"])
    X.append(df["p1"])
    X.append(df["p2"])
    X.append(df["p3"])
    X=np.array(X).T
    y=np.array(df["m_eva"])
    
    X, y = sklearn.utils.shuffle(X, y) 
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = test_size, random_state = 0)
    return X_train, X_test, y_train, y_test



    
def predict_rf2(df):
    clf = load('./models/RF-2.model')
    X=[]
    for fea in FEATURES_NAME:
        X.append(df[fea])
    X.append(df["ai_eva"])
    X.append(df["p1"])
    X.append(df["p2"])
    X.append(df["p3"])
    X=np.array(X).T
    y=np.array(df["m_eva"])
    
    prediction = clf.predict(X)
    pred_prob=clf.predict_proba(X)
    
    df.insert(df.shape[1], "ai2_eva", prediction)
    df.insert(df.shape[1], "ai2_p1", pred_prob[:,0])
    df.insert(df.shape[1], "ai2_p2", pred_prob[:,1])
    df.insert(df.shape[1], "ai2_p3", pred_prob[:,2])
 
    return df





if __name__ == "__main__":
    FILEPATH = sys.argv[1]
    df = pd.read_csv(FILEPATH+"/fea_m_compare.csv") 
    # df = pd.read_csv(FILEPATH+"/fea_m_all.csv") 
    df.attack_effi[df.attack_effi>0.05] = 0.05

    X_train, X_test, y_train, y_test=prepare_Xy(df)
    X=np.concatenate((X_train, X_test))
    y=np.concatenate((y_train, y_test))

    max_depth=5
    n_estimators=50
    min_samples_leaf=4

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features="log2", random_state=1)
    clf.fit(X, y)
    acc = clf.score(X, y)
    dump(clf, './models/RF-2.model')

    df_new=predict_rf2(df) 
    df_new.to_csv(FILEPATH+"/fea_m_ai2.csv", encoding="utf_8_sig", index=False)
