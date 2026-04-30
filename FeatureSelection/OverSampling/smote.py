#!/usr/bin/env python
# coding: utf-8

from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import FeatureSelection.OverSampling.functions as functions
import functions

# Load Dataset
df = functions.load_data()


# stores cpg markers
headers = df.columns
headers = headers[:-1] # get rid of target
headers



for i in range(1, 11):
    final_list = []
    final_list2 = []
    
    # Apply SMOTE Algorithm
    X = df.iloc[:,:-1].values
    y = df.target.values
    smote = SMOTE(k_neighbors=3, sampling_strategy=0.5, random_state=i) # changing the SMOTE rand state
    X_new, y_new = smote.fit_resample(X, y)
    
    smote_df = pd.DataFrame(data=X_new, columns=headers)
    smote_df["Target"] = y_new
    smote_df = smote_df.sample(frac = 1)
    target_names = {
   "normal":0,
    "tumor":1, 
    }

    smote_df['is_tumor'] = smote_df['Target'].map(target_names)
    smote_df = smote_df.drop("Target", axis=1)
    X = smote_df.iloc[:,1:-1] 
    y = smote_df.iloc[:,-1] 

    print("Only random forest")
    sel = RandomForestClassifier(n_estimators = 500, random_state=0)
    sel.fit(X, y)
    rf_sel_features=sel.feature_importances_
    #Some features are not important and get marked as 0. Hence we will extract features with importance > 0
        
    feat_importances = pd.Series(rf_sel_features, index=X.columns)
    print(feat_importances)
    
    list1 = []
    for j in range(len(feat_importances.index)):
        if feat_importances.values[j]>0:
            list1.append(feat_importances.index[j])
    print(len(list1))

    final_list.append(list1)
    my_df = pd.DataFrame(final_list)
    my_df = my_df.T 
    file_name = 'output_files2/Meth/Meth_Impt_Features' + str(i) + 'RF.csv'
    my_df.to_csv(file_name, index=False, header=False)
    
  
    print("Only ANOVA")
    X_new = SelectKBest(f_classif, k=X.shape[1]).fit_transform(X, y)
    fvals, pvals = f_classif(X_new, y)
    #Get the list of cpg marker names
    col_list =  X.columns.tolist()
    #verify the the fvals are same as total markers
    print(len(fvals))
    #Check how many markers are less than 0.05
    h = sum(float(num) < 0.05 for num in pvals)
    print(h)


    #create a list with marker names haiing p-values less than 0.05
    list1 = []
    for j in range(len(pvals)):
        if pvals[j] < 0.05:
            list1.append(col_list[j])
    print(len(list1))
    final_list2.append(list1)
    
    my_df2 = pd.DataFrame(final_list2)
    my_df2 = my_df2.T 
    file_name2 = 'output_files2/Meth/Meth_Impt_Features' + str(i) + 'Anova.csv'
    my_df2.to_csv(file_name2, index=False, header=False)

