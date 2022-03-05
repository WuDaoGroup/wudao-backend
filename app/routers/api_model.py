import os
import xgboost as xgb

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import column_or_1d

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal


router = APIRouter(prefix = "/models")

@router.post("/predict/svc")
def SVC( filename: str = Form(...), percent: str = Form(...) ):
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float(percent)
    clf = svm.SVC()
    
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    y = column_or_1d(y, warn=True).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
    
    clf.fit( X_train, y_train )
    
    per_test_data = clf.score( X_test, y_test )
    # .astype('int')
    per_test_data = format(per_test_data, '.4f')
    print(per_test_data)

    res["result_accuracyOfTestData"] = per_test_data
    res["code"] = """ 
    def SVC( filename: str = Form(...), percent: str = Form(...) ):
        data = pd.read_csv('./static/data/'+ filename)
        per = float(percent)
        clf = svm.SVC()
        
        X = data.iloc[:, 1:]
        y = data.iloc[:, :1]
        y = column_or_1d(y, warn=True).ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
        
        clf.fit( X_train, y_train )
        
        per_test_data = clf.score( X_test, y_test )
        per_test_data = format(per_test_data, '.4f')
    """
    return res

@router.post("/predict/xgboost")
def xgboost( filename: str = Form(...), percent: str = Form(...) ):
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float( percent )
    
    xlf = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=10, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, )
    
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=per, random_state=0 )
    
    xlf.fit( X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)], early_stopping_rounds = 100 )
    print( "!!!!!", xlf.score(X_test, y_test) )
    per_train_data = xlf.score(X_test, y_test)
    per_train_data = format(per_train_data, '.4f')
    res["result_accuracyOfTestData"] = per_train_data
    res["code"] = """
    def xgboost( filename: str = Form(...), percent: str = Form(...) ):
        data = pd.read_csv('./static/data/'+ filename)
        per = float( percent )
        
        xlf = xgb.XGBRegressor(max_depth=10, 
                            learning_rate=0.1, 
                            n_estimators=10, 
                            silent=True, 
                            objective='reg:linear', 
                            nthread=-1, 
                            gamma=0,
                            min_child_weight=1, 
                            max_delta_step=0, 
                            subsample=0.85, 
                            colsample_bytree=0.7, 
                            colsample_bylevel=1, 
                            reg_alpha=0, 
                            reg_lambda=1, 
                            scale_pos_weight=1, 
                            seed=1440, )
        
        X = data.iloc[:, 1:]
        y = data.iloc[:, :1]
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=per, random_state=0 )
        
        xlf.fit( X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)], early_stopping_rounds = 100 )
        per_train_data = xlf.score(X_test, y_test)
        per_train_data = format(per_train_data, '.4f')
    """
    return res

@router.post("/predict/lassoLars")
def lasso_lars( filename: str = Form(...), alpha: str = Form(...), normalize: str = Form(...), 
    percent: str = Form(...) ):
    i = 0
    normal = True
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    al = float( alpha )
    per = float( percent )
    if normalize == 'True':
        normal = True
    else:
        normal = False
    
    reg = linear_model.LassoLars(alpha = al, normalize = normal)
    
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
    
    reg.fit( X_train, y_train )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()
    # print(reg_list1, reg_list2)

    per_train_data = reg.score(X_train, y_train)
    print("per_train_data:", per_train_data)
    per_train_data = format(per_train_data, '.4f')
    

    per_test_data = reg.score(X_test, y_test)
    print("per_test_data:", per_test_data)
    per_test_data = format(per_test_data, '.4f')
    

    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1):
        reg_list1[i] = float(reg_list1[i])
        reg_list1[i] = format(reg_list1[i], '.4f')
        i += 1
    res["result_coef"] = reg_list1
    res["result_intercept"] = reg_list2[0]
    res["result_accuracyOfTestData"] = per_test_data
    res["result_accuracyOfTrainData"] = per_train_data
    res["code"] = """
    def lasso_lars( filename: str = Form(...), alpha: str = Form(...), normalize: str = Form(...), 
        percent: str = Form(...) ):
        i = 0
        normal = True
        data = pd.read_csv('./static/data/'+ filename)
        al = float( alpha )
        per = float( percent )
        if normalize == 'True':
            normal = True
        else:
            normal = False
        
        reg = linear_model.LassoLars(alpha = al, normalize = normal)
        
        X = data.iloc[:, 1:]
        y = data.iloc[:, :1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
        
        reg.fit( X_train, y_train )
        reg_list1 = reg.coef_.tolist()
        reg_list2 = reg.intercept_.tolist()

        per_train_data = reg.score(X_train, y_train)
        print("per_train_data:", per_train_data)
        per_train_data = format(per_train_data, '.4f')

        per_test_data = reg.score(X_test, y_test)
        print("per_test_data:", per_test_data)
        per_test_data = format(per_test_data, '.4f')

        reg_list2[0] = format(reg_list2[0], '.4f') 
        while i < len(reg_list1):
            reg_list1[i] = float(reg_list1[i])
            reg_list1[i] = format(reg_list1[i], '.4f')
            i += 1
    """
    return res

@router.post("/predict/ols")
def ordinary_least_squares( filename: str = Form(...), percent: str = Form(...) ):
    i = 0
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float(percent)
    reg = linear_model.LinearRegression()
    
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
    
    reg.fit( X_train, y_train )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()
    print(reg_list1, reg_list2)

    per_train_data = reg.score(X_train, y_train)
    # print("per_train_data:", per_train_data)
    per_train_data = format(per_train_data, '.4f')

    per_test_data = reg.score(X_test, y_test)
    # print("per_test_data:", per_test_data)
    per_test_data = format(per_test_data, '.4f')
    
    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    res["result_accuracyOfTestData"] = per_test_data
    res["result_accuracyOfTrainData"] = per_train_data
    res["code"] = """
    def ordinary_least_squares( filename: str = Form(...), percent: str = Form(...) ):
        i = 0
        data = pd.read_csv('./static/data/'+ filename)
        per = float(percent)
        reg = linear_model.LinearRegression()
        
        X = data.iloc[:, 1:]
        y = data.iloc[:, :1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
        
        reg.fit( X_train, y_train )
        reg_list1 = reg.coef_.tolist()
        reg_list2 = reg.intercept_.tolist()

        per_train_data = reg.score(X_train, y_train)
        per_train_data = format(per_train_data, '.4f')

        per_test_data = reg.score(X_test, y_test)
        per_test_data = format(per_test_data, '.4f')
        
        reg_list2[0] = format(reg_list2[0], '.4f') 
        while i < len(reg_list1[0]):
            reg_list1[0][i] = format(reg_list1[0][i], '.4f')
            i += 1
    """
    return res

@router.post("/predict/lasso")
def lasso( filename: str = Form(...), alpha: str = Form(...), percent: str = Form(...)):
    i = 0
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float(percent)
    alpha = float(alpha)
    reg = linear_model.Ridge(alpha)
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
    reg.fit( X_train, y_train )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()

    per_train_data = reg.score(X_train, y_train)
    # print("per_train_data:", per_train_data)
    per_train_data = format(per_train_data, '.4f')

    per_test_data = reg.score(X_test, y_test)
    # print("per_test_data:", per_test_data)
    per_test_data = format(per_test_data, '.4f')

    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    res["result_accuracyOfTestData"] = per_test_data
    res["result_accuracyOfTrainData"] = per_train_data
    res["code"] = """
    def lasso( filename: str = Form(...), alpha: str = Form(...), percent: str = Form(...)):
        i = 0
        data = pd.read_csv('./static/data/'+ filename)
        per = float(percent)
        alpha = float(alpha)
        reg = linear_model.Ridge(alpha)
        X = data.iloc[:, 1:]
        y = data.iloc[:, :1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
        reg.fit( X_train, y_train )
        reg_list1 = reg.coef_.tolist()
        reg_list2 = reg.intercept_.tolist()

        per_train_data = reg.score(X_train, y_train)
        per_train_data = format(per_train_data, '.4f')

        per_test_data = reg.score(X_test, y_test)
        per_test_data = format(per_test_data, '.4f')

        reg_list2[0] = format(reg_list2[0], '.4f') 
        while i < len(reg_list1[0]):
            reg_list1[0][i] = format(reg_list1[0][i], '.4f')
            i += 1
    """
    return res

@router.post("/predict/ridge_regression")
def ridge_regression( filename: str = Form(...), alpha: str = Form(...), percent: str = Form(...)):
    i = 0
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float(percent)
    alpha = float(alpha)
    reg = linear_model.Ridge(alpha)
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
    reg.fit( X_train, y_train )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()

    per_train_data = reg.score(X_train, y_train)
    # print("per_train_data:", per_train_data)
    per_train_data = format(per_train_data, '.4f')

    per_test_data = reg.score(X_test, y_test)
    # print("per_test_data:", per_test_data)
    per_test_data = format(per_test_data, '.4f')
    
    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    res["result_accuracyOfTestData"] = per_test_data
    res["result_accuracyOfTrainData"] = per_train_data
    res["code"] = """
    def ridge_regression( filename: str = Form(...), alpha: str = Form(...), percent: str = Form(...)):
        i = 0
        data = pd.read_csv('./static/data/'+ filename)
        per = float(percent)
        alpha = float(alpha)
        reg = linear_model.Ridge(alpha)
        X = data.iloc[:, 1:]
        y = data.iloc[:, :1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
        reg.fit( X_train, y_train )
        reg_list1 = reg.coef_.tolist()
        reg_list2 = reg.intercept_.tolist()

        per_train_data = reg.score(X_train, y_train)
        per_train_data = format(per_train_data, '.4f')

        per_test_data = reg.score(X_test, y_test)
        per_test_data = format(per_test_data, '.4f')
        
        reg_list2[0] = format(reg_list2[0], '.4f') 
        while i < len(reg_list1[0]):
            reg_list1[0][i] = format(reg_list1[0][i], '.4f')
            i += 1
    """
    return res

@router.post("/predict/bdtr")
def boosted_decision_tree_regression( filename: str = Form(...) ):
    img_addr = './static/images/' + filename + '_img.png'
    print(filename)
    res = {}
    data = pd.read_csv('./static/data/'+ filename) 
    rng = np.random.RandomState(1)
    X = data.iloc[:, 1:2].values
    y = data.iloc[:, 0:1].values
    regr_1 = DecisionTreeRegressor(max_depth=4)
    regr_2 = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
    )
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    plt.figure()
    plt.scatter(X, y, c="k", label="training samples")
    plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
    plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.savefig(img_addr)
    res['pic_addr'] = filename + '_img.png'
    res["code"] = """
    def boosted_decision_tree_regression( filename: str = Form(...) ):
        img_addr = './static/images/' + filename + '_img.png'
        print(filename)
        data = pd.read_csv('./static/data/'+ filename) 
        rng = np.random.RandomState(1)
        X = data.iloc[:, 1:2].values
        y = data.iloc[:, 0:1].values
        regr_1 = DecisionTreeRegressor(max_depth=4)
        regr_2 = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
        )
        regr_1.fit(X, y)
        regr_2.fit(X, y)
        y_1 = regr_1.predict(X)
        y_2 = regr_2.predict(X)
        plt.figure()
        plt.scatter(X, y, c="k", label="training samples")
        plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
        plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Boosted Decision Tree Regression")
        plt.legend()
        plt.savefig(img_addr)
    """
    return res