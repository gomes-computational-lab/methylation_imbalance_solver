#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


# In[15]:


# get dataset
file = "../../../Final/Main Code/Preprocessing/Methylation_Imputation/BetaData_SimpleImpute_Zero.csv"
df = pd.read_csv(file, sep=",")

df = df.drop("Donor_Sample", axis=1)

df.head()


# In[23]:


normal_list = df.index[df['is_tumor'] == 0].tolist()
print(normal_list)
tumor_list = df.index[df['is_tumor'] == 1].tolist()
print(tumor_list)


# In[25]:


# Instead of taking the tumor in sequence, pull them randomly and remove them from the list, 
# so they cannot be reused. Redo the RUS with this. 

# Might be a random subset function I can use, so I don’t have to create them on my own.
# Use a different seed each time I randomly subset.

# read the "HERE"


# In[26]:


#Divide the list of 179 tumor in groups of 8 and merge them with 4 normal samples. 
# 8 times 22 = 176
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import random

def run_RUS(index):
    tumor_list_cpy = []
    tumor_list_cpy = tumor_list_cpy + tumor_list


    final_list1 = []
    final_list2 = []

    for i in range(0, 176, 8):
        tumor_subset = []

         # changing the random seed each iteration
        random.seed(i)

        # randomly subsetting the tumors
        for _ in range(8):
            rand_index = random.choice(tumor_list_cpy)
            tumor_list_cpy.remove(rand_index)
            tumor_subset.append(rand_index)

        # combine the 8 random tumors with the 4 normals
        res = sorted(tumor_subset + normal_list) 
        df_selected = df.iloc[res]

        #split dataset into features and target
        X = df_selected.iloc[:,:-1] 
        Y = df_selected.iloc[:,-1] 
        print("Y = ",Y.shape)
             
    
        print("Only random forest")
        sel = RandomForestClassifier(n_estimators = 500, random_state=index)
        sel.fit(X, Y)
        rf_sel_features=sel.feature_importances_
        #Some features are not important and get marked as 0. Hence we will extract features with importance > 0

        feat_importances = pd.Series(rf_sel_features, index=X.columns)

        list1 = []
        for j in range(len(feat_importances.index)):
            if feat_importances.values[j]>0:
                list1.append(feat_importances.index[j])
        print(len(list1))

        final_list1.append(list1)

        print("Only ANOVA")
        X_new = SelectKBest(f_classif, k=X.shape[1]).fit_transform(X, Y)
        fvals, pvals = f_classif(X_new, Y)

        #Get the list of cpg marker names
        col_list =  X.columns.tolist()

        #Check how many markers are less than 0.05
        h = sum(float(num) < 0.05 for num in pvals)

        #create a list with marker names haiing p-values less than 0.05
        list2 = []
        for j in range(len(pvals)):
            if pvals[j] < 0.05:
                list2.append(col_list[j])
        print(len(list2))
        final_list2.append(list2)

        # combine all lists into one
        big_list1 = []
        big_list1 = big_list1 + final_list1[0]
        for i in range(1,len(final_list1)):
            big_list1.extend(final_list1[i])
        my_df = pd.DataFrame(big_list1, columns=[["cpg_marker"]])
        file_name = 'output_files2/Meth/Meth_Impt_Features_RF_'+ str(index) + '.csv'
        my_df.to_csv(file_name, index=False, header=True)

        big_list2 = []
        big_list2 = big_list2 + final_list2[0]
        for i in range(1,len(final_list2)):
            big_list2.extend(final_list2[i])
        my_df2 = pd.DataFrame(big_list2, columns=[["cpg_marker"]])
        file_name = 'output_files2/Meth/Meth_Impt_Features_ANOVA_'+ str(index) + '.csv'
        my_df2.to_csv(file_name, index=False, header=True)


for i in range(1, 11):
    run_RUS(i)

