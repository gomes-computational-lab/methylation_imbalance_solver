#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from imblearn.under_sampling import AllKNN


# In[2]:


# get dataset
file = "../../../Final/Main Code/Preprocessing/Methylation_Imputation/BetaData_SimpleImpute_Zero.csv"
meth_df = pd.read_csv(file, sep=",")

meth_df = meth_df.drop("Donor_Sample", axis=1)

meth_df.head()


# In[3]:


# stores ensg id
headers = meth_df.columns
headers = headers[:-1] # get rid of target
headers


# In[4]:


# from collections import Counter

# X = rna_df.iloc[:,:-1].values
# y = rna_df.target.values

# # visualize data
# counter = Counter(y)
# print(counter)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def run_AllKNN(index):
    
    final_list1 = []
    final_list2 = []

    for i in range(1, 23):
        X = meth_df.iloc[:,:-1].values
        y = meth_df.is_tumor.values

        allknn = AllKNN(n_neighbors=i)
        X_res, y_res = allknn.fit_resample(X, y)

        knn_df = pd.DataFrame(data=X_res, columns=headers)
        knn_df["is_tumor"] = y_res
        knn_df = knn_df.sample(frac = 1)
        knn_df

        X = knn_df.iloc[:,1:-1] 
        Y = knn_df.iloc[:,-1] 

        print("Only random forest")
        sel = RandomForestClassifier(n_estimators = 500, random_state=index)
        sel.fit(X, Y)
        rf_sel_features=sel.feature_importances_
        #Some features are not important and get marked as 0. Hence we will extract features with importance > 0

        feat_importances = pd.Series(rf_sel_features, index=X.columns)
        print(feat_importances)

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
        
    final_flat1 = [item for sublist in final_list1 for item in sublist]
    my_df = pd.DataFrame(final_flat1)
    my_df = my_df.T 
    file_name = 'output_files2/Meth/Meth_Impt_Features' + str(index) + 'RF.csv'
    my_df.to_csv(file_name, index=False, header=False)
    
    final_flat2 = [item for sublist in final_list2 for item in sublist]
    my_df2 = pd.DataFrame(final_flat2)
    my_df2 = my_df2.T 
    file_name2 = 'output_files2/Meth/Meth_Impt_Features' + str(index) + 'Anova.csv'
    my_df2.to_csv(file_name2, index=False, header=False)


# In[ ]:


for i in range(1, 11):
    print(i)
    run_AllKNN(i)


# In[ ]:


# # combine all lists into one
# big_list1 = []
# big_list1 = big_list1 + final_list1[0]

# print(len(big_list1))

# for i in range(1,len(final_list1)):
#     big_list1.extend(final_list1[i])
    
# print(len(big_list1))


# In[ ]:


# # combine all lists into one
# big_list2 = []
# big_list2 = big_list2 + final_list2[0]

# print(len(big_list2))

# for i in range(1,len(final_list2)):
#     big_list2.extend(final_list2[i])
    
# print(len(big_list2))


# In[ ]:


# # final_list holds the output from each iteration [[itr. 1],[itr. 2],[itr. 3], etc]
# my_df1 = pd.DataFrame(big_list1, columns=[["cpg_marker"]])
# my_df1.to_csv('output_files/Meth_Impt_Features_RF_LOOPED.csv', index=False, header=True)


# In[ ]:


# # final_list holds the output from each iteration [[itr. 1],[itr. 2],[itr. 3], etc]
# my_df2 = pd.DataFrame(big_list2, columns=[["cpg_marker"]])
# my_df2.to_csv('output_files/Meth_Impt_Features_ANOVA_LOOPED.csv', index=False, header=True)

