##########################################
####### Multilabel Classification ########
##########################################

# load required packages
import pandas as pd
from sklearn import preprocessing
import numpy as np

# import dataset and make sure that it contains a binary variable for each of the C diseases (here: IMID, FMS, OA, OPCD)

# get one hot encoding for categorical variables
one_hot = pd.get_dummies(data[["Admission_Type", "Prednisolon_total", "Sex"]])
data = data.join(one_hot)

# split dataset in features and target variable
features_cols = ["Admission_Type_outpatient", "Admission_Type_inpatient", "Age", "Sex_male", "Sex_female", "Duration_Symptoms_Diagnosis_1", "Tender_Points", "Sum_FIRST", "CRP", "Prednisolon_total_no", "Prednisolon_total_yes", "Prednisolon_Dose", "NRS_Pain"]
target_cols = ["FMS", "OPCD", "OA", "IMID"]

X = data[features_cols] # features
y = data[target_cols] # targets




###### Classifier Chain ########

from sklearn.multioutput import ClassifierChain



###### Decision Tree ########

# load required packages
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# define grid search
params_dt = {
    "estimator__criterion" : ["gini", "entropy"],
    "estimator__max_depth": range(3,8),
    "estimator__min_samples_split": range(5,10)
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
    cc_dt = ClassifierChain(DecisionTreeClassifier(random_state = 664), random_state = 664)
    
    # fit model
    grid_dt = GridSearchCV(estimator = cc_dt,
                           param_grid = params_dt,
                           cv = inner_cv,
                           scoring = 'f1_macro',
                           refit = True).fit(X_train, y_train)
    
    # get best model fit on whole training data
    best_model = grid_dt.best_estimator_
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # save model
    models_dt.append(best_model)
    
    # derive binary variable with true values for IMID
    y_test = y_test["IMID"]
    pred = pred[:, 3]

    # evaluate the model
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()        
    sensitivity = tp / (tp + fn)        
    specificity = tn / (tn + fp)        
    precision = tp / (tp + fp)            
    f1 = (2*tp) / (2*tp + fp + fn)
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
params_rf = {
    "estimator__n_estimators": [10, 25, 50 , 100, 250],
    "estimator__criterion" : ["gini", "entropy"],
    "estimator__max_depth": range(3,8),
    "estimator__min_samples_split": range(5,10)
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
    cc_rf = ClassifierChain(RandomForestClassifier(random_state = 54986), random_state = 54986)
    
    # fit model
    grid_rf = GridSearchCV(estimator = cc_rf,
                           param_grid = params_rf,
                           cv = inner_cv,
                           scoring = 'f1_macro',
                           refit = True).fit(X_train, y_train)
    
    # get best model fit on whole training data
    best_model = grid_rf.best_estimator_
    
    # save model
    models_rf.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # derive binary variable with true values for IMID
    y_test = y_test["IMID"]
    pred = pred[:, 3]

    # evaluate the model
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()        
    sensitivity = tp / (tp + fn)        
    specificity = tn / (tn + fp)        
    precision = tp / (tp + fp)            
    f1 = (2*tp) / (2*tp + fp + fn)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_rf = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]





###### Logistic Regression ########

# load required packages
from sklearn.linear_model import LogisticRegression

# define grid search
params_lr = {
    "estimator__penalty" : ["l2", None],
    "estimator__tol" : [0.0001, 0.0005, 0.001],
    "estimator__solver" : ["lbfgs", "newton-cg", "saga", "sag"]
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
    cc_lr = ClassifierChain(LogisticRegression(random_state = 6865, max_iter = 10000), random_state = 6865)
    
    # fit model
    grid_lr = GridSearchCV(estimator = cc_lr,
                           param_grid = params_lr,
                           cv = inner_cv,
                           scoring = 'f1_macro',
                           refit = True).fit(X_train, y_train)
    
    # get best model fit on whole training data
    best_model = grid_lr.best_estimator_
    
    # save model
    models_lr.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # derive binary variable with true values for IMID
    y_test = y_test["IMID"]
    pred = pred[:, 3]

    # evaluate the model
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()        
    sensitivity = tp / (tp + fn)        
    specificity = tn / (tn + fp)        
    precision = tp / (tp + fp)            
    f1 = (2*tp) / (2*tp + fp + fn)
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
params_mlp = {
    "estimator__hidden_layer_sizes": [(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100)],
    "estimator__solver": ["lbfgs", "sgd", "adam"],
    "estimator__activation": ["logistic", "relu"],
    "estimator__alpha": [0.0001, 0.0005, 0.001],
    "estimator__learning_rate": ["constant", "invscaling", "adaptive"],
    "estimator__learning_rate_init": [0.0001, 0.001, 0.01]
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
    cc_mlp = ClassifierChain(MLPClassifier(random_state = 1164, max_iter = 10000), random_state = 1164)
    
    # fit model
    grid_mlp = GridSearchCV(estimator = cc_mlp,
                           param_grid = params_mlp,
                           cv = inner_cv,
                           scoring = 'f1_macro',
                           refit = True).fit(X_train, y_train)
    
    # get best model fit on whole training data
    best_model = grid_mlp.best_estimator_
    
    # save model
    models_mlp.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # derive binary variable with true values for IMID
    y_test = y_test["IMID"]
    pred = pred[:, 3]

    # evaluate the model
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()        
    sensitivity = tp / (tp + fn)        
    specificity = tn / (tn + fp)        
    precision = tp / (tp + fp)            
    f1 = (2*tp) / (2*tp + fp + fn)
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
params_knn = {
    "estimator__n_neighbors": [3, 5, 15, 25]
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
    cc_knn = ClassifierChain(KNeighborsClassifier())
    
    # fit model
    grid_knn = GridSearchCV(estimator = cc_knn,
                           param_grid = params_knn,
                           cv = inner_cv,
                           scoring = 'f1_macro',
                           refit = True).fit(X_train, y_train)
    
    # get best model fit on whole training data
    best_model = grid_knn.best_estimator_
    
    # save model
    models_knn.append(best_model)
    
    # evaluate model on whole test set
    pred = best_model.predict(X_test)
    
    # derive binary variable with true values for IMID
    y_test = y_test["IMID"]
    pred = pred[:, 3]

    # evaluate the model
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()        
    sensitivity = tp / (tp + fn)        
    specificity = tn / (tn + fp)        
    precision = tp / (tp + fp)            
    f1 = (2*tp) / (2*tp + fp + fn)
    metrics = [sensitivity, specificity, precision, f1]
     
    # store result
    results_outer.loc[len(results_outer)] = metrics
    
# reset index of data frame
results_outer = results_outer.reset_index(drop = True)    
    
# summarize the estimated performance of the model
mean_metrics_knn = [results_outer["Sensitivity"].mean(), results_outer["Specificity"].mean(), results_outer["Precision"].mean(), results_outer["F1-Score"].mean()]
