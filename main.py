#Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, make_scorer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import preprocessing 

#Loading kaggle dataset
!kaggle competitions download -c diamonds-datamad1020-rev
!tar -xzvf diamonds-datamad1020-rev.zip

#Modeling, Prediction, and Evaluation
X = train_data.drop(['price'],1)
y = train_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Linear Regression
lr = LinearRegression()
model = lr.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE: {}".format(mean_squared_error((y_test),(y_pred))))
print("R2  : {}".format(np.sqrt(r2_score((y_test),(y_pred)))))

#Random Forest
rr  = RandomForestRegressor()
rr.fit(X_train,y_train)
y_pred = rr.predict(X_test)

print("MSE: {}".format(mean_squared_error((y_test),(y_pred))))
print("R2  : {}".format(np.sqrt(r2_score((y_test),(y_pred)))))

#Using Grid Search CV 
param_grid = {
'n_estimators': [400,600, 800,1000],
 'min_samples_split': [2,3,4],
 'min_samples_leaf': [2,3,4],
 'max_depth': [4, 5],
'warm_start': [False]}

MSE = make_scorer(mean_squared_error)

grid_search = GridSearchCV(estimator = rr, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 1, scoring=MSE)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

param_grid = {
'n_estimators': [1000],
 'min_samples_split': [2],
 'min_samples_leaf': [3],
 'max_depth': [4],
'warm_start': [False]}
#Running RF model again with best parameters found using Grid Search CV
grid_search2 = GridSearchCV(estimator = rr, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 1, scoring=MSE)                             
grid_search2.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

#Using Grid Search 2.0
n_estimators = [int(x) for x in np.linspace(10,200,10)]
max_depth = [int(x) for x in np.linspace(10,100,10)]
min_samples_split = [2,3,4,5]
min_samples_leaf = [2,3,4,5]
random_grid = {'n_estimators':n_estimators,'max_depth':max_depth,
               'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}

rr = RandomForestRegressor()
rr_random = RandomizedSearchCV(estimator=rr,
                               param_distributions=random_grid,
                               cv = 3)

rr_random.fit(X_train,y_train)
y_pred = rr_random.predict(X_test)

print("MSE: {}".format(mean_squared_error((y_test),(y_pred))))
print("R2  : {}".format(np.sqrt(r2_score((y_test),(y_pred)))))

rr_random.best_params_
rf = RandomForestRegressor(n_estimators=94,
                         min_samples_split=5,
                         min_samples_leaf=2,
                         max_depth=90)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

print("MSE: {}".format(mean_squared_error((y_test),(y_pred))))
print("R2  : {}".format(np.sqrt(r2_score((y_test),(y_pred)))))

