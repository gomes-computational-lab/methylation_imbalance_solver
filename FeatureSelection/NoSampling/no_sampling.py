#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[2]:


# get dataset
file = "../../Final/Main Code/Preprocessing/Methylation_Imputation/BetaData_SimpleImpute_Zero.csv"
meth_df = pd.read_csv(file, sep=",")

target_names = {
    0:"normal",
    1:"tumor", 
}

meth_df['target'] = meth_df['is_tumor'].map(target_names)
meth_df = meth_df.drop("is_tumor", axis=1)
meth_df = meth_df.drop("Donor_Sample", axis=1)


# In[ ]:


# stores cpg markers
headers = meth_df.columns
headers = headers[:-1] # get rid of target
headers


# In[ ]:


from collections import Counter

X = meth_df.iloc[:,:-1]
y = meth_df.iloc[:,-1] 

# visualize data
counter = Counter(y)
print(counter)


# In[ ]:


# Running the random forest 10 times, changing random_state every time
for i in range(1, 11):
    
    final_list = []

    print("Only random forest")
    sel = RandomForestClassifier(n_estimators = 500, random_state=i)
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
    file_name = 'output_files/Random_Forest_Looped/Meth_Impt_Features' + str(i) + 'RF.csv'
    my_df.to_csv(file_name, index=False, header=False)


# In[ ]:


# my_df = pd.DataFrame(final_list)
# my_df = my_df.T 
# file_name = 'output_files/Meth_Impt_Features_RF.csv'
# my_df.to_csv(file_name, index=False, header=False)


# In[ ]:


# Run Random Forest 10 times with different random states - Anova is fine only being run once

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
print(h)


#create a list with marker names haiing p-values less than 0.05
list1 = []
for j in range(len(pvals)):
    if pvals[j] < 0.05:
        list1.append(col_list[j])
print(len(list1))
final_list2.append(list1)


# In[ ]:


my_df2 = pd.DataFrame(final_list2)
my_df2 = my_df2.T 
file_name2 = 'output_files/Meth_Impt_Features_Anova.csv'
my_df2.to_csv(file_name2, index=False, header=False)

