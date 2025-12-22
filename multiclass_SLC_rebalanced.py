################################################################
####### Multiclass SLC Models with Class Rebalancing ###########
################################################################

# load required packages
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import resample
import numpy as np

# import dataset and make sure that it contains one variable indicating the disease (categorical variable with C categories, here: 4 categories)

# get one hot encoding for categorical variables
one_hot = pd.get_dummies(data[["Admission_Type", "Prednisolon_total", "Sex"]])
data = data.join(one_hot)

# encode target variable
le = preprocessing.LabelEncoder() # define encoder
le.fit(['Fibromyalgia', 'Immune-mediated Inflammatory Disease', 'Osteoarthritis', 'other pain-causing diseases']) # fit encoder
data["disease"] = le.transform(data["disease"]) # transform target variable

# check for class imbalance
data["disease"].value_counts()

# split dataset in features and target variable
features_cols = ["Admission_Type_outpatient", "Admission_Type_inpatient", "Age", "Sex_male", "Sex_female", "Duration_Symptoms_Diagnosis_1", "Tender_Points", "Sum_FIRST", "CRP", "Prednisolon_total_no", "Prednisolon_total_yes", "Prednisolon_Dose", "NRS_Pain"]
target_cols = ["disease"]

X = data[features_cols] # features
y = data[target_cols] # target





###### Decision Tree ########

# load required packages
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# define grid search
param_dict_dt = {
    "criterion" : ["gini", "entropy"],
    "max_depth": range(3,8),
    "min_samples_split": range(5,10),
}
  
# specify inner and outer loop
inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 761)
outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 57843)

# create empty list for results
results_outer = pd.DataFrame(columns = ["Sensitivity", "Specificity", "Precision", "F1-Score"])
models_dt = []

# outer loop
for train_ix, test_ix in outer_cv.split(X):
    
    # train-test-split
    train, X_test = data.iloc[train_ix], X.iloc[test_ix]
    y_test = y.iloc[test_ix]
    
    # get mean of classes
    mean = train["disease"].value_counts().mean()
    mean = int(mean)
    
    # define which values need to be up-/downsampled
    counts = train["disease"].value_counts()
    above = []
    below = []
    for element in counts:
        if element >= mean:
            above.append(counts[counts == element].index.values.astype(int)[0])
        else:
            below.append(counts[counts == element].index.values.astype(int)[0])
            
    # get data sets with samples to be up- or downsampled
    majority = train[train["disease"].isin(above)]
    minority = train[train["disease"].isin(below)]
    
    # save disease numbers as list
    disease_majority = majority["disease"].unique()
    disease_minority = minority["disease"].unique()
    
    # downsampling
    majority_down = []
    for disease_nr in disease_majority:
        majority_down.append(resample(majority[majority["disease"] == disease_nr],
                                 replace = False,
                                 n_samples = mean,
                                 random_state = 343))
    
    majority_down = pd.concat(majority_down)
    
    # upsampling
    minority_up = []
    for disease_nr in disease_minority:
        count = sum(minority["disease"] == disease_nr)
        diff = mean - count
        minority_up.append(resample(minority[minority["disease"] == disease_nr],
                                    replace = True,
                                    n_samples = diff,
                                    random_state = 6428))
    
    minority_up = pd.concat(minority_up)
    minority_up = pd.concat([minority_up, minority])
                                            
    # combine rebalanced data sets
    data_rebalanced = pd.concat([minority_up, majority_down])
    data_rebalanced["disease"].value_counts()
    
    # split train data into features and target
    X_train = data_rebalanced[features_cols] # features
    y_train = data_rebalanced[target_cols] # target
    y_train = np.array(y_train)
        
    # normalize data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # classifier
    dt = DecisionTreeClassifier(random_state = 664)
    
    # fit model
    grid_dt = GridSearchCV(estimator = dt,
                             param_grid = param_dict_dt,
                             cv = inner_cv,
                             scoring = 'f1_macro',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_dt.best_estimator_
    
    # save model
    models_dt.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)

    # derive binary variable with true values for IMID
    y_test = y_test.to_numpy()
    y_test_binary = np.where(y_test == 1, 1, 0)

    # derive result variable for IMID
    pred_binary = np.where(pred == 1, 1, 0)
    
    # evaluate the model
    f1 = f1_score(y_test_binary, pred_binary)
    sensitivity = recall_score(y_test_binary, pred_binary)
    precision = precision_score(y_test_binary, pred_binary)
    specificity = recall_score(y_test_binary, pred_binary, pos_label = 0)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_dt = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]





###### Random Forest ########

# load required packages
from sklearn.ensemble import RandomForestClassifier

# define grid search
param_dict_rf = {
    "criterion" : ["gini", "entropy"],
    "max_depth": range(3,8),
    "min_samples_split": range(5,10),
    "n_estimators" : [10, 25, 50 , 100, 250],
}
  
# specify inner and outer loop
inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 761)
outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 57843)

# create empty list for results
results_outer = pd.DataFrame(columns = ["Sensitivity", "Specificity", "Precision", "F1-Score"])
models_rf = []

# outer loop
for train_ix, test_ix in outer_cv.split(X):
    
    # train-test-split
    train, X_test = data.iloc[train_ix], X.iloc[test_ix]
    y_test = y.iloc[test_ix]
    
    # get mean of classes
    mean = train["disease"].value_counts().mean()
    mean = int(mean)
    
    # define which values need to be up-/downsampled
    counts = train["disease"].value_counts()
    above = []
    below = []
    for element in counts:
        if element >= mean:
            above.append(counts[counts == element].index.values.astype(int)[0])
        else:
            below.append(counts[counts == element].index.values.astype(int)[0])
            
    # get data sets with samples to be up- or downsampled
    majority = train[train["disease"].isin(above)]
    minority = train[train["disease"].isin(below)]
    
    # save disease numbers as list
    disease_majority = majority["disease"].unique()
    disease_minority = minority["disease"].unique()
    
    # downsampling
    majority_down = []
    for disease_nr in disease_majority:
        majority_down.append(resample(majority[majority["disease"] == disease_nr],
                                 replace = False,
                                 n_samples = mean,
                                 random_state = 343))
    
    majority_down = pd.concat(majority_down)
    
    # upsampling
    minority_up = []
    for disease_nr in disease_minority:
        count = sum(minority["disease"] == disease_nr)
        diff = mean - count
        minority_up.append(resample(minority[minority["disease"] == disease_nr],
                                    replace = True,
                                    n_samples = diff,
                                    random_state = 6428))
    
    minority_up = pd.concat(minority_up)
    minority_up = pd.concat([minority_up, minority])
                                            
    # combine rebalanced data sets
    data_rebalanced = pd.concat([minority_up, majority_down])
    data_rebalanced["disease"].value_counts()
    
    # split train data into features and target
    X_train = data_rebalanced[features_cols] # features
    y_train = data_rebalanced[target_cols] # target
    y_train = np.array(y_train)
        
    # normalize data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # classifier
    rf = RandomForestClassifier(random_state = 54986)
    
    # fit model
    grid_rf = GridSearchCV(estimator = rf,
                             param_grid = param_dict_rf,
                             cv = inner_cv,
                             scoring = 'f1_macro',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_rf.best_estimator_
    
    # save model
    models_rf.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)

    # derive binary variable with true values for IMID
    y_test = y_test.to_numpy()
    y_test_binary = np.where(y_test == 1, 1, 0)

    # derive result variable for IMID
    pred_binary = np.where(pred == 1, 1, 0)
    
    # evaluate the model
    f1 = f1_score(y_test_binary, pred_binary)
    sensitivity = recall_score(y_test_binary, pred_binary)
    precision = precision_score(y_test_binary, pred_binary)
    specificity = recall_score(y_test_binary, pred_binary, pos_label = 0)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_rf = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]





###### Logistic Regression Model ######

# load required packages
from sklearn.linear_model import LogisticRegression

# define grid search
param_dict_lr = {
    "penalty" : ["l2", None],
    "tol" : [0.0001, 0.0005, 0.001],
    "solver" : ["lbfgs", "newton-cg", "saga", "sag"]
    }
  
# specify inner and outer loop
inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 761)
outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 57843)

# create empty list for results
results_outer = pd.DataFrame(columns = ["Sensitivity", "Specificity", "Precision", "F1-Score"])
models_lr = []

# outer loop
for train_ix, test_ix in outer_cv.split(X):
    
    # train-test-split
    train, X_test = data.iloc[train_ix], X.iloc[test_ix]
    y_test = y.iloc[test_ix]
    
    # get mean of classes
    mean = train["disease"].value_counts().mean()
    mean = int(mean)
    
    # define which values need to be up-/downsampled
    counts = train["disease"].value_counts()
    above = []
    below = []
    for element in counts:
        if element >= mean:
            above.append(counts[counts == element].index.values.astype(int)[0])
        else:
            below.append(counts[counts == element].index.values.astype(int)[0])
            
    # get data sets with samples to be up- or downsampled
    majority = train[train["disease"].isin(above)]
    minority = train[train["disease"].isin(below)]
    
    # save disease numbers as list
    disease_majority = majority["disease"].unique()
    disease_minority = minority["disease"].unique()
    
    # downsampling
    majority_down = []
    for disease_nr in disease_majority:
        majority_down.append(resample(majority[majority["disease"] == disease_nr],
                                 replace = False,
                                 n_samples = mean,
                                 random_state = 343))
    
    majority_down = pd.concat(majority_down)
    
    # upsampling
    minority_up = []
    for disease_nr in disease_minority:
        count = sum(minority["disease"] == disease_nr)
        diff = mean - count
        minority_up.append(resample(minority[minority["disease"] == disease_nr],
                                    replace = True,
                                    n_samples = diff,
                                    random_state = 6428))
    
    minority_up = pd.concat(minority_up)
    minority_up = pd.concat([minority_up, minority])
                                            
    # combine rebalanced data sets
    data_rebalanced = pd.concat([minority_up, majority_down])
    data_rebalanced["disease"].value_counts()
    
    # split train data into features and target
    X_train = data_rebalanced[features_cols] # features
    y_train = data_rebalanced[target_cols] # target
    y_train = np.array(y_train)
        
    # normalize data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # classifier
    lr = LogisticRegression(random_state = 6865, max_iter = 10000)
    
    # fit model
    grid_lr = GridSearchCV(estimator = lr,
                             param_grid = param_dict_lr,
                             cv = inner_cv,
                             scoring = 'f1_macro',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_lr.best_estimator_
    
    # save model
    models_lr.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)

    # derive binary variable with true values for IMID
    y_test = y_test.to_numpy()
    y_test_binary = np.where(y_test == 1, 1, 0)

    # derive result variable for IMID
    pred_binary = np.where(pred == 1, 1, 0)
    
    # evaluate the model
    f1 = f1_score(y_test_binary, pred_binary)
    sensitivity = recall_score(y_test_binary, pred_binary)
    precision = precision_score(y_test_binary, pred_binary)
    specificity = recall_score(y_test_binary, pred_binary, pos_label = 0)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_lr = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]





###### Neural Network ########

# load required packages
from sklearn.neural_network import MLPClassifier

# define grid search
param_dict_mlp = {
    "hidden_layer_sizes": [(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100)],
    "solver": ["lbfgs", "sgd", "adam"],
    "activation": ["logistic", "relu"],
    "alpha": [0.0001, 0.0005, 0.001],
    "learning_rate": ["constant", "invscaling", "adaptive"]
}
  
# specify inner and outer loop
inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 761)
outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 57843)

# create empty list for results
results_outer = pd.DataFrame(columns = ["Sensitivity", "Specificity", "Precision", "F1-Score"])
models_mlp = []

# outer loop
for train_ix, test_ix in outer_cv.split(X):
    
    # train-test-split
    train, X_test = data.iloc[train_ix], X.iloc[test_ix]
    y_test = y.iloc[test_ix]
    
    # get mean of classes
    mean = train["disease"].value_counts().mean()
    mean = int(mean)
    
    # define which values need to be up-/downsampled
    counts = train["disease"].value_counts()
    above = []
    below = []
    for element in counts:
        if element >= mean:
            above.append(counts[counts == element].index.values.astype(int)[0])
        else:
            below.append(counts[counts == element].index.values.astype(int)[0])
            
    # get data sets with samples to be up- or downsampled
    majority = train[train["disease"].isin(above)]
    minority = train[train["disease"].isin(below)]
    
    # save disease numbers as list
    disease_majority = majority["disease"].unique()
    disease_minority = minority["disease"].unique()
    
    # downsampling
    majority_down = []
    for disease_nr in disease_majority:
        majority_down.append(resample(majority[majority["disease"] == disease_nr],
                                 replace = False,
                                 n_samples = mean,
                                 random_state = 343))
    
    majority_down = pd.concat(majority_down)
    
    # upsampling
    minority_up = []
    for disease_nr in disease_minority:
        count = sum(minority["disease"] == disease_nr)
        diff = mean - count
        minority_up.append(resample(minority[minority["disease"] == disease_nr],
                                    replace = True,
                                    n_samples = diff,
                                    random_state = 6428))
    
    minority_up = pd.concat(minority_up)
    minority_up = pd.concat([minority_up, minority])
                                            
    # combine rebalanced data sets
    data_rebalanced = pd.concat([minority_up, majority_down])
    data_rebalanced["disease"].value_counts()
    
    # split train data into features and target
    X_train = data_rebalanced[features_cols] # features
    y_train = data_rebalanced[target_cols] # target
    y_train = np.array(y_train)
        
    # normalize data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # classifier
    mlp = MLPClassifier(random_state = 1164, max_iter=10000)
    
    # fit model
    grid_mlp = GridSearchCV(estimator = mlp,
                             param_grid = param_dict_mlp,
                             cv = inner_cv,
                             scoring = 'f1_macro',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_mlp.best_estimator_
    
    # save model
    models_mlp.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)

    # derive binary variable with true values for IMID
    y_test = y_test.to_numpy()
    y_test_binary = np.where(y_test == 1, 1, 0)

    # derive result variable for IMID
    pred_binary = np.where(pred == 1, 1, 0)
    
    # evaluate the model
    f1 = f1_score(y_test_binary, pred_binary)
    sensitivity = recall_score(y_test_binary, pred_binary)
    precision = precision_score(y_test_binary, pred_binary)
    specificity = recall_score(y_test_binary, pred_binary, pos_label = 0)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_mlp = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]





###### kNN ########

# load required packages
from sklearn.neighbors import KNeighborsClassifier

# define grid search
param_dict_knn = {
    "n_neighbors": [3, 5, 15, 25]
}
  
# specify inner and outer loop
inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 761)
outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 57843)

# create empty list for results
results_outer = pd.DataFrame(columns = ["Sensitivity", "Specificity", "Precision", "F1-Score"])
models_knn = []

# outer loop
for train_ix, test_ix in outer_cv.split(X):
    
    # train-test-split
    train, X_test = data.iloc[train_ix], X.iloc[test_ix]
    y_test = y.iloc[test_ix]
    
    # get mean of classes
    mean = train["disease"].value_counts().mean()
    mean = int(mean)
    
    # define which values need to be up-/downsampled
    counts = train["disease"].value_counts()
    above = []
    below = []
    for element in counts:
        if element >= mean:
            above.append(counts[counts == element].index.values.astype(int)[0])
        else:
            below.append(counts[counts == element].index.values.astype(int)[0])
            
    # get data sets with samples to be up- or downsampled
    majority = train[train["disease"].isin(above)]
    minority = train[train["disease"].isin(below)]
    
    # save disease numbers as list
    disease_majority = majority["disease"].unique()
    disease_minority = minority["disease"].unique()
    
    # downsampling
    majority_down = []
    for disease_nr in disease_majority:
        majority_down.append(resample(majority[majority["disease"] == disease_nr],
                                 replace = False,
                                 n_samples = mean,
                                 random_state = 343))
    
    majority_down = pd.concat(majority_down)
    
    # upsampling
    minority_up = []
    for disease_nr in disease_minority:
        count = sum(minority["disease"] == disease_nr)
        diff = mean - count
        minority_up.append(resample(minority[minority["disease"] == disease_nr],
                                    replace = True,
                                    n_samples = diff,
                                    random_state = 6428))
    
    minority_up = pd.concat(minority_up)
    minority_up = pd.concat([minority_up, minority])
                                            
    # combine rebalanced data sets
    data_rebalanced = pd.concat([minority_up, majority_down])
    data_rebalanced["disease"].value_counts()
    
    # split train data into features and target
    X_train = data_rebalanced[features_cols] # features
    y_train = data_rebalanced[target_cols] # target
    y_train = np.array(y_train)
        
    # normalize data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # classifier
    knn = KNeighborsClassifier()
    
    # fit model
    grid_knn = GridSearchCV(estimator = knn,
                             param_grid = param_dict_knn,
                             cv = inner_cv,
                             scoring = 'f1_macro',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_knn.best_estimator_
    
    # save model
    models_knn.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)

    # derive binary variable with true values for IMID
    y_test = y_test.to_numpy()
    y_test_binary = np.where(y_test == 1, 1, 0)

    # derive result variable for IMID
    pred_binary = np.where(pred == 1, 1, 0)
    
    # evaluate the model
    f1 = f1_score(y_test_binary, pred_binary)
    sensitivity = recall_score(y_test_binary, pred_binary)
    precision = precision_score(y_test_binary, pred_binary)
    specificity = recall_score(y_test_binary, pred_binary, pos_label = 0)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_knn = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]