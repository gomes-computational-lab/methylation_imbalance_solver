#!/usr/bin/env python
# coding: utf-8

# IMPORTS
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import functions


# Load Dataset
df = functions.load_data()


# stores ensg id
headers = df.columns
headers = headers[:-1] # get rid of target


for i in range(1, 11):
    final_list = []
    final_list2 = []
    
    # Create a RandomState instance
    rs_instance = np.random.RandomState(i*7)  # You can use any integer seed

    # Apply ADAYSN Algorithm
    X = df.iloc[:,:-1].values
    y = df.target.values
    ada = ADASYN(random_state=rs_instance, n_neighbors=3, sampling_strategy=0.5)
    X_new, y_new = ada.fit_resample(X, y)
      
    X, y = functions.shuffle_data(X_new, y_new, headers)

    # Apply Random Forest Feature Selection
    rf_features = functions.get_rf_features(X,y)
    final_list.append(rf_features)
    rf_df = pd.DataFrame(final_list)
    rf_df = rf_df.T 
    file_name = 'output_files2/Meth/Meth_Impt_Features' + str(i) + 'RF.csv'
    rf_df.to_csv(file_name, index=False, header=False)
    
    # Apply ANOVA Feature Selection
    anova_features = functions.get_anova_features(X,y)
    final_list2.append(anova_features)
    anova_df = pd.DataFrame(final_list2)
    anova_df = anova_df.T 
    file_name2 = 'output_files2/Meth/Meth_Impt_Features' + str(i) + 'Anova.csv'
    anova_df.to_csv(file_name2, index=False, header=False)




