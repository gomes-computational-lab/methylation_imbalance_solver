#!/usr/bin/env python
# coding: utf-8

# IMPORTS
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import functions


# Load Dataset
df = functions.load_data()

# stores cpg markers
headers = df.columns
headers = headers[:-1] # get rid of target
headers


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


for i in range(1, 11):
    final_list = []
    
    # Create a RandomState instance
    rs_instance = np.random.RandomState(i*7)  # You can use any integer seed

    # Use it in RandomOverSampler
    X = df.iloc[:,:-1].values
    y = df.target.values
    ros = RandomOverSampler(sampling_strategy=0.5, random_state=rs_instance)
    X_new, y_new = ros.fit_resample(X, y)
      
    ros_df = pd.DataFrame(data=X_new, columns=headers)
    ros_df["Target"] = y_new
    ros_df = ros_df.sample(frac = 1)
    target_names = {
   "normal":0,
    "tumor":1, 
    }

    ros_df['is_tumor'] = ros_df['Target'].map(target_names)
    ros_df = ros_df.drop("Target", axis=1)
    X = ros_df.iloc[:,1:-1] 
    y = ros_df.iloc[:,-1] 

    print("Only random forest")
    sel = RandomForestClassifier(n_estimators = 500, random_state=0)
    sel.fit(X, y)
    rf_sel_features=sel.feature_importances_
    #Some features are not important and get marked as 0. Hence we will extract features with importance > 0
        
    feat_importances = pd.Series(rf_sel_features, index=X.columns)
    
    list1 = []
    for j in range(len(feat_importances.index)):
        if feat_importances.values[j]>0:
            list1.append(feat_importances.index[j])

    final_list.append(list1)
    my_df = pd.DataFrame(final_list)
    my_df = my_df.T 
    file_name = 'output_files2/Meth/Meth_Impt_Features' + str(i) + 'RF.csv'
    my_df.to_csv(file_name, index=False, header=False)
    
    final_list2 = []
    print("Only ANOVA")
    X_new = SelectKBest(f_classif, k=X.shape[1]).fit_transform(X, y)
    fvals, pvals = f_classif(X_new, y)
    #Get the list of cpg marker names
    col_list =  X.columns.tolist()
    #verify the the fvals are same as total markers
    print(len(fvals))
    #Check how many markers are less than 0.05
    h = sum(float(num) < 0.05 for num in pvals)


    #create a list with marker names haiing p-values less than 0.05
    list1 = []
    for j in range(len(pvals)):
        if pvals[j] < 0.05:
            list1.append(col_list[j])
    final_list2.append(list1)
    
    my_df2 = pd.DataFrame(final_list2)
    my_df2 = my_df2.T 
    file_name2 = 'output_files2/Meth/Meth_Impt_Features' + str(i) + 'Anova.csv'
    my_df2.to_csv(file_name2, index=False, header=False)


# In[ ]:




