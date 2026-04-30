#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from collections import Counter
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans

# In[2]:

# get dataset
file = "../../../Final/Main Code/Preprocessing/Methylation_Imputation/BetaData_SimpleImpute_Zero.csv"
meth_df = pd.read_csv(file, sep=",")

meth_df = meth_df.drop("Donor_Sample", axis=1)

#meth_df.head()


# In[3]:


# stores ensg id
headers = meth_df.columns
headers = headers[:-1] # get rid of target


X = meth_df.iloc[:,:-1].values
y = meth_df.target.values

# visualize data
counter = Counter(y)
print(counter)


def run_CC(index):

    final_list1 = list()
    final_list2 = list()

    # doing 8 iterations to ensure all majority class data is being used (176/184 ish) bc 11 normal
    # however, we can't guarentee that KMeans doesn't reuse some of the same majority samples 
    # but that is why we do 10 rounds of this to account for randomness :)
    for i in range(0,8):
        X = meth_df.iloc[:,:-1].values
        y = meth_df.is_tumor.values

        # setting KMeans random_state=None makes it use the random init which will be different every time ran
        cc = ClusterCentroids(sampling_strategy=0.5, estimator=KMeans(n_init=1, random_state=None), random_state=i) 
        X_res, y_res = cc.fit_resample(X, y)

        cc_df = pd.DataFrame(data=X_res, columns=headers)
        cc_df["is_tumor"] = y_res
        cc_df = cc_df.sample(frac = 1)

        X = cc_df.iloc[:,1:-1] 
        y = cc_df.iloc[:,-1] 

        print("Only random forest")
        sel = RandomForestClassifier(n_estimators = 500, random_state=i)
        sel.fit(X, y)
        rf_sel_features=sel.feature_importances_
        #Some features are not important and get marked as 0. Hence we will extract features with importance > 0

        feat_importances = pd.Series(rf_sel_features, index=X.columns)

        list1 = []
        for j in range(len(feat_importances.index)):
            if feat_importances.values[j]>0:
                list1.append(feat_importances.index[j])

        final_list1.append(list1)
        
        print("Only ANOVA")
        X_new = SelectKBest(f_classif, k=X.shape[1]).fit_transform(X, y)
        fvals, pvals = f_classif(X_new, y)

        #Get the list of cpg marker names
        col_list =  X.columns.tolist()

        #Check how many markers are less than 0.05
        h = sum(float(num) < 0.05 for num in pvals)

        #create a list with marker names haiing p-values less than 0.05
        list2 = []
        for j in range(len(pvals)):
            if pvals[j] < 0.05:
                list2.append(col_list[j])
        final_list2.append(list2)
    
    final_flat1 = [item for sublist in final_list1 for item in sublist]
    my_df = pd.DataFrame(final_flat1)
    my_df = my_df.T 
    file_name = 'output_files2/Meth_Actual/Meth_Impt_Features' + str(index) + 'RF.csv'
    my_df.to_csv(file_name, index=False, header=False)
    
    final_flat2 = [item for sublist in final_list2 for item in sublist]
    my_df2 = pd.DataFrame(final_flat2)
    my_df2 = my_df2.T 
    file_name2 = 'output_files2/Meth_Actual/Meth_Impt_Features' + str(index) + 'Anova.csv'
    my_df2.to_csv(file_name2, index=False, header=False)



for i in range(1, 10):
    print(i)
    run_CC(i)
