import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

def load_data():
    file = "../../../Final/Main Code/Preprocessing/Methylation_Imputation/BetaData_SimpleImpute_Zero.csv"
    df = pd.read_csv(file, sep=",")

    target_names = {
     0:"normal",
     1:"tumor", 
        }

    df['target'] = df['is_tumor'].map(target_names)
    df = df.drop("is_tumor", axis=1)
    df = df.drop("Donor_Sample", axis=1)

    return df

def shuffle_data(X_new, y_new, headers):
    df = pd.DataFrame(data=X_new, columns=headers)
    df["Target"] = y_new
    df = df.sample(frac = 1)
    target_names = {
   "normal":0,
    "tumor":1, 
    }

    df['is_tumor'] = df['Target'].map(target_names)
    df = df.drop("Target", axis=1)
    X = df.iloc[:,1:-1] 
    y = df.iloc[:,-1] 
    return X, y
    
def get_rf_features(X, y):
    model = RandomForestClassifier(n_estimators=500, random_state=0)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)

    selected = list(importances[importances > 0].index)
     #Some features are not important and get marked as 0. Hence we will extract features with importance > 0
        
    return selected

def get_anova_features(X, y, threshold=0.05):
    pvals = f_classif(X, y)

    cols = X.columns.tolist()

    selected = [
        cols[i]
        for i in range(len(pvals))
        if pvals[i] < threshold
    ]

    return selected