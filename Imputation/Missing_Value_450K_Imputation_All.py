#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
rng = np.random.RandomState(0)

from sklearn.ensemble import RandomForestRegressor

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from pandas import read_csv, DataFrame, concat
import matplotlib.pyplot as plt


# In[2]:


# load dataset
url = '../../../../Data_Code/CSV_Files/BetaData_AllRounded.csv'
dataframe = read_csv(url, header=0, na_values='')#.iloc[:, 1:]
print(dataframe)
print('The size of dataframe is: ', dataframe.shape)

#Get headers
headers = read_csv(url, nrows=0).columns.tolist()
print('Header length is: '+str(len(headers)) + ' and first value is : '+str(headers[0]))

#Get sample names
samples = dataframe.iloc[:,:1]
print(samples.shape)
print(samples)


# In[3]:


#Function to calculate the NA percentages and return them for plotting
def calculate_NA_Percentage(df):
    print(df.shape[1])
    distribution = []
    distribution_perc = []
    # summarize the number of rows with missing values for each column
    for i in range(df.shape[1]):
        col_name = df.columns[i]
        # count number of rows with missing values
        n_miss = df[col_name].isnull().sum()
        perc = round(n_miss / df.shape[0] * 100, 2)
        print('> %s, Missing: %d (%.1f%%)' % (col_name, n_miss, perc))
        distribution.append(n_miss)
        distribution_perc.append(perc)

    return distribution, distribution_perc


#Create a dictionary of missing values
def plotNA_Distribution(NaN_List, NaN_P):
    bins = 10
    fig, ax = plt.subplots(figsize =(12, 8))
    arr = ax.hist(NaN_List)
    for i in range(bins):
        plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
    print('##############################################')
    fig, ax = plt.subplots(figsize =(12, 8))
    arr = ax.hist(NaN_P)
    for i in range(bins):
        plt.text(arr[1][i],arr[0][i],str(arr[0][i]))


# In[4]:


#Find NA distribution
NaN_Distribution_1, NaN_Percent = calculate_NA_Percentage(dataframe.iloc[:,1:])


# In[5]:


#Plot the NA distribution before NaN removal
plotNA_Distribution(NaN_Distribution_1, NaN_Percent)


# In[6]:


# Delete columns containing 30% missing values
percent = 30.0
min_count =  int(((100-percent)/100)*dataframe.shape[0] + 1) #Keep rows with at least the min_counnt non-NA values

dataframe_NaN_Removed = dataframe.dropna(axis=1, thresh=min_count)
#Find headers of new removed CpG dataframe.
#Get headers
headers_rm = dataframe_NaN_Removed.columns.tolist()
NaN_Distribution_2, NaN_Percent_2 = calculate_NA_Percentage(dataframe_NaN_Removed.iloc[:,1:])
print('Header length is: '+str(len(headers_rm)) + ' and first value is : '+str(headers_rm[0]))
print(dataframe.shape[0])
print(min_count)
print(dataframe_NaN_Removed)


# In[7]:


#Plot the NA distribution after NaN removal
plotNA_Distribution(NaN_Distribution_2, NaN_Percent_2)


# In[8]:


#Get CpG values
CpGs = dataframe_NaN_Removed.iloc[:,1:]
print(CpGs.shape)
print(CpGs)


# In[9]:


# split into input and output elements using is_tumor column as dependent variable
data = CpGs.values
# print(data)
X, y = data[:, :-1], data[:, -1]
print(X)
print('##########')
print(X.shape)
print('###############')
print(y)
print('##########')
print(y.shape)

##############################
##############################


# In[10]:


N_SPLITS = 5
regressor = RandomForestRegressor(random_state=0)

def get_scores_for_imputer(imputer, X_missing, y_missing):
    estimator = make_pipeline(imputer, regressor)
    impute_scores = cross_val_score(estimator, X_missing, y_missing,
                                    scoring='neg_mean_squared_error',
                                    cv=N_SPLITS)
    return impute_scores


x_labels = ['Zero imputation',
            'KNN Imputation',
            'Mean Imputation']

mses_data = np.zeros(3)
stds_data = np.zeros(3)


# In[11]:


#Impute with zero
def get_impute_zero_score(X_missing, y_missing):

    imputer = SimpleImputer(missing_values=np.nan, add_indicator=False,
                            strategy='constant', fill_value=0)
    # print total missing
    print('Missing before imputation: %d' % sum(np.isnan(X_missing).flatten()))
    imputer.fit(X_missing)
    # transform the dataset
    Xtrans = imputer.transform(X_missing)
    # print total missing
    print('Missing after imputation: %d' % sum(np.isnan(Xtrans).flatten()))

    zero_impute_scores = get_scores_for_imputer(imputer, X_missing, y_missing)
    return zero_impute_scores.mean(), zero_impute_scores.std(), Xtrans


mses_data[0], stds_data[0], Xtrans_Zero = get_impute_zero_score(X, y)


# In[12]:


print(Xtrans_Zero)
#Merge back the dependent and independent variables.
y_new_zero = np.expand_dims(y, axis = 0)
samples_new_zero = np.concatenate((Xtrans_Zero, y_new_zero.T), axis=1)
print(samples_new_zero.shape)
print('#######################')
#Add the column headers
print(headers_rm[1:])
df_imputed_zero = DataFrame(samples_new_zero, columns = headers_rm[1:])
print(df_imputed_zero)
print('########################')
imputed_df_sample_zero = concat([dataframe['Donor_Sample'], df_imputed_zero.set_index(dataframe.index)], axis=1)
print(imputed_df_sample_zero)
imputed_df_sample_zero = imputed_df_sample_zero.sort_index()
imputed_df_sample_zero.to_csv('BetaData_SimpleImpute_Zero.csv', sep=',')
print(imputed_df_sample_zero)


# In[13]:


#Impute with KNN
def get_impute_knn_score(X_missing, y_missing):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=False)

    # print total missing
    print('Missing before KNN imputation: %d' % sum(np.isnan(X_missing).flatten()))
    imputer.fit(X_missing)
    # transform the dataset
    Xtrans = imputer.transform(X_missing)
    # print total missing
    print('Missing after KNN imputation: %d' % sum(np.isnan(Xtrans).flatten()))


    knn_impute_scores = get_scores_for_imputer(imputer, X_missing, y_missing)
    return knn_impute_scores.mean(), knn_impute_scores.std(), Xtrans


mses_data[1], stds_data[1], XTrans_KNN = get_impute_knn_score(X, y)


# In[14]:


print(XTrans_KNN)
#Merge back the dependent and independent variables.
y_new_KNN = np.expand_dims(y, axis = 0)
samples_new_KNN = np.concatenate((XTrans_KNN, y_new_KNN.T), axis=1)
print(samples_new_KNN.shape)
print('#######################')
#Add the column headers
print(headers_rm[1:])
df_imputed_KNN = DataFrame(samples_new_KNN, columns = headers_rm[1:])
print(df_imputed_KNN)
print('########################')
imputed_df_sample_KNN = concat([dataframe['Donor_Sample'], df_imputed_KNN.set_index(dataframe.index)], axis=1)
print(imputed_df_sample_KNN)
imputed_df_sample_KNN = imputed_df_sample_KNN.sort_index()
imputed_df_sample_KNN.to_csv('BetaData_SimpleImpute_KNN.csv', sep=',')
print(imputed_df_sample_KNN)


# In[15]:


#Impute with mean
def get_impute_mean(X_missing, y_missing):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean",
                            add_indicator=False)

    # print total missing
    print('Missing before mean imputation: %d' % sum(np.isnan(X_missing).flatten()))
    imputer.fit(X_missing)
    # transform the dataset
    Xtrans = imputer.transform(X_missing)
    # print total missing
    print('Missing after mean imputation: %d' % sum(np.isnan(Xtrans).flatten()))

    mean_impute_scores = get_scores_for_imputer(imputer, X_missing, y_missing)
    return mean_impute_scores.mean(), mean_impute_scores.std(), Xtrans


mses_data[2], stds_data[2], Xtrans_mean = get_impute_mean(X, y)


# In[16]:


print(Xtrans_mean)
#Merge back the dependent and independent variables.
y_new_mean = np.expand_dims(y, axis = 0)
samples_new_mean = np.concatenate((Xtrans_mean, y_new_mean.T), axis=1)
print(samples_new_mean.shape)
print('#######################')
#Add the column headers
print(headers_rm[1:])
df_imputed_mean = DataFrame(samples_new_mean, columns = headers_rm[1:])
print(df_imputed_mean)
print('########################')
imputed_df_sample_mean = concat([dataframe['Donor_Sample'], df_imputed_mean.set_index(dataframe.index)], axis=1)
print(imputed_df_sample_mean)
imputed_df_sample_mean = imputed_df_sample_mean.sort_index()
imputed_df_sample_mean.to_csv('BetaData_SimpleImpute_Mean.csv', sep=',')
print(imputed_df_sample_mean)


# In[17]:


# #Iterative imputation
# def get_impute_iterative(X_missing, y_missing):
#     imputer = IterativeImputer(missing_values=np.nan, add_indicator=False,
#                                random_state=0, n_nearest_features=5,
#                                sample_posterior=True)

#     # print total missing
#     print('Missing before mean imputation: %d' % sum(np.isnan(X_missing).flatten()))
#     imputer.fit(X_missing)
#     # transform the dataset
#     Xtrans = imputer.transform(X_missing)
#     # print total missing
#     print('Missing after mean imputation: %d' % sum(np.isnan(Xtrans).flatten()))

#     iterative_impute_scores = get_scores_for_imputer(imputer,
#                                                      X_missing,
#                                                      y_missing)
#     return iterative_impute_scores.mean(), iterative_impute_scores.std(), Xtrans


# mses_data[3], stds_data[3], Xtrans_iterative  = get_impute_iterative(X,y)


# mses_data = mses_data * -1


# In[18]:


# print(Xtrans_iterative)
# #Merge back the dependent and independent variables.
# y_new_iterative = np.expand_dims(y, axis = 0)
# samples_new_iterative = np.concatenate((Xtrans_iterative, y_new_iterative.T), axis=1)
# print(samples_new_iterative.shape)
# print('#######################')
# #Add the column headers
# print(headers_rm[1:])
# df_imputed_iterative = DataFrame(samples_new_iterative, columns = headers_rm[1:])
# print(df_imputed_iterative)
# print('########################')
# imputed_df_sample_iterative = concat([dataframe['Donor_Sample'], df_imputed_iterative.set_index(dataframe.index)], axis=1)
# print(imputed_df_sample_iterative)
# imputed_df_sample_iterative = imputed_df_sample_iterative.sort_index()
# imputed_df_sample_iterative.to_csv('/content/drive/MyDrive/Bioinformatics/BetaData_27K_SimpleImpute_Iterative_1.csv', sep=',')
# print(imputed_df_sample_iterative)


# In[19]:


#Plot the results
import matplotlib.pyplot as plt


n_bars = len(mses_data)
xval = np.arange(n_bars)

colors = ['r', 'g', 'b']

# plot dataset results
ax2 = plt.subplot(122)
for j in xval:
    ax2.barh(j, mses_data[j], xerr=stds_data[j],
             color=colors[j], alpha=0.6, align='center')

ax2.set_title('Imputation Techniques with 27K dataset')
ax2.set_yticks(xval)
ax2.set_xlabel('MSE')
ax2.invert_yaxis()
ax2.set_yticklabels([''] * n_bars)

plt.show()

# You can also try different techniques. For instance, the median is a more
# robust estimator for data with high magnitude variables which could dominate
# results (otherwise known as a 'long tail').


# In[20]:


# x_labels = ['Zero imputation',
#             'KNN Imputation',
#             'Mean Imputation']


print(mses_data)
print('######')
print(stds_data)
print('########')


# In[ ]:




