#####################################
####### Binary SLC Models ###########
#####################################

# load required packages
import pandas as pd
from sklearn import preprocessing
import numpy as np

# import dataset and make sure that it contains a binary variable for the disease of interest (here: IMID)

# get one hot encoding for categorical variables
one_hot = pd.get_dummies(data[["Admission_Type", "Prednisolon_total", "Sex"]])
data = data.join(one_hot)

# split dataset into features and target variable
features_cols = ["Admission_Type_outpatient", "Admission_Type_inpatient", "Age", "Sex_male", "Sex_female", "Duration_Symptoms_Diagnosis_1", "Tender_Points", "Sum_FIRST", "CRP", "Prednisolon_total_no", "Prednisolon_total_yes", "Prednisolon_Dose", "NRS_Pain"]
target_cols = ["IMID"]

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
for train_id, test_id in outer_cv.split(X):
    
    # train-test-split
    X_train, X_test = X.iloc[train_id], X.iloc[test_id]
    y_train, y_test = y.iloc[train_id], y.iloc[test_id]
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
                             scoring = 'f1',
                             refit = True).fit(X_train, y_train)
    
    # get best model fit on whole training data
    best_model = grid_dt.best_estimator_
    
    # save model
    models_dt.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # evaluate the model
    f1 = f1_score(y_test, pred)
    sensitivity = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    specificity = recall_score(y_test, pred, pos_label = 0)
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
for train_id, test_id in outer_cv.split(X):
    
    # train-test-split
    X_train, X_test = X.iloc[train_id], X.iloc[test_id]
    y_train, y_test = y.iloc[train_id], y.iloc[test_id]
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
                             scoring = 'f1',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_rf.best_estimator_
    
    # save model
    models_rf.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # evaluate the model
    f1 = f1_score(y_test, pred)
    sensitivity = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    specificity = recall_score(y_test, pred, pos_label = 0)
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
for train_id, test_id in outer_cv.split(X):
    
    # train-test-split
    X_train, X_test = X.iloc[train_id], X.iloc[test_id]
    y_train, y_test = y.iloc[train_id], y.iloc[test_id]
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
                             scoring = 'f1',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_lr.best_estimator_
    
    # save model
    models_lr.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # evaluate the model
    f1 = f1_score(y_test, pred)
    sensitivity = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    specificity = recall_score(y_test, pred, pos_label = 0)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_lr = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]





###### Multilayer Perceptron ########

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
for train_id, test_id in outer_cv.split(X):
    
    # train-test-split
    X_train, X_test = X.iloc[train_id], X.iloc[test_id]
    y_train, y_test = y.iloc[train_id], y.iloc[test_id]
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
                             scoring = 'f1',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_mlp.best_estimator_
    
    # save model
    models_mlp.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # evaluate the model
    f1 = f1_score(y_test, pred)
    sensitivity = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    specificity = recall_score(y_test, pred, pos_label = 0)
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
for train_id, test_id in outer_cv.split(X):
    
    # train-test-split
    X_train, X_test = X.iloc[train_id], X.iloc[test_id]
    y_train, y_test = y.iloc[train_id], y.iloc[test_id]
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
                             scoring = 'f1',
                             refit = True).fit(X_train, y_train.ravel())
    
    # get best model fit on whole training data
    best_model = grid_knn.best_estimator_
    
    # save model
    models_knn.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # evaluate the model
    f1 = f1_score(y_test, pred)
    sensitivity = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    specificity = recall_score(y_test, pred, pos_label = 0)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_knn = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]