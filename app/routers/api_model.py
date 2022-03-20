import os
import math
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_recall_curve 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal


router = APIRouter(prefix = "/models")

#分类分析
@router.post("/predict/xgboost-classification")
def xgboost_classification( filename: str = Form(...), percent: str = Form(...) ):
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float( percent )
    auprc = 1
    list = []
    precision = []
    recall = []
    thresholds = []
    
    xlf = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,         # 树的个数--1000棵树建立xgboost
        max_depth=6,               # 树的深度
        min_child_weight = 1,      # 叶子节点最小权重
        gamma=0.,                  # 惩罚项中叶子结点个数前的参数
        subsample=0.8,             # 随机选择80%样本建立决策树
        colsample_btree=0.8,       # 随机选择80%特征建立决策树
        objective='multi:softmax', # 指定损失函数
        scale_pos_weight=1,        # 解决样本个数不平衡的问题
        random_state=27,           # 随机数
        num_class= 2
    )
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=per, random_state=0 )
    
    xlf.fit(
        X_train,
        y_train,
        eval_set = [(X_test,y_test)],
        eval_metric = "mlogloss",
        early_stopping_rounds = 10,
        verbose = True
    )
    
    #accuracy
    y_pred = xlf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    #per_test_data
    per_test_data = xlf.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')
    #AUROC
    clf_ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    auroc = roc_auc_score(y, clf_.predict_proba(X)[:, 1], multi_class='ovo')

    #判断有几个预测的目标
    for i in y:
        if i  == 0 or i == 1 or i == -1:
            if i not in list:
                list.append(i)
        else:
            auprc = 0
            break
    if len(list) != 2:
        auprc = 0
    if 0 in list and -1 in list:
        auprc = 0
    #AUPRC
    if auprc == 1:
        clf__ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
        y_score = clf__.decision_function( X )
        precision, recall, thresholds = precision_recall_curve( y, y_score )
        precision = precision.tolist() 
        recall = recall.tolist() 
        thresholds = thresholds.tolist()

    res["auroc"] = auroc
    res["recall"] = recall
    res["accuracy"] = accuracy
    res["precision"] = precision
    res["thresholds"] = thresholds
    res["result_accuracy_of_test_data"] = per_test_data
    res["code"] = """
def xgboost_classification( filename: str = Form(...), percent: str = Form(...) ):
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float( percent )
    auprc = 1
    
    xlf = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,         # 树的个数--1000棵树建立xgboost
        max_depth=6,               # 树的深度
        min_child_weight = 1,      # 叶子节点最小权重
        gamma=0.,                  # 惩罚项中叶子结点个数前的参数
        subsample=0.8,             # 随机选择80%样本建立决策树
        colsample_btree=0.8,       # 随机选择80%特征建立决策树
        objective='multi:softmax', # 指定损失函数
        scale_pos_weight=1,        # 解决样本个数不平衡的问题
        random_state=27            # 随机数
    )
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=per, random_state=0 )
    
    xlf.fit(
        X_train,
        y_train,
        eval_set = [(X_test,y_test)],
        eval_metric = "mlogloss",
        early_stopping_rounds = 10,
        verbose = True
    )
    
    #accuracy
    y_pred = xlf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    #per_train_data
    per_train_data = xlf.score(X_train, y_train)
    per_train_data = format(per_train_data, '.4f')
    #per_test_data
    per_test_data = xlf.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')
    #AUROC
    clf_ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    auroc = roc_auc_score(y, clf_.predict_proba(X)[:, 1], multi_class='ovo')

    #判断有几个预测的目标
    for i in y:
        if i  == 0 or i == 1 or i == -1:
            if i not in list:
                list.append(i)
        else:
            auprc = 0
            break
    if len(list) != 2:
        auprc = 0
    if 0 in list and -1 in list:
        auprc = 0
    #AUPRC
    if auprc == 1:
        clf__ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
        y_score = clf__.decision_function( X )
        precision, recall, thresholds = precision_recall_curve( y, y_score )
        precision = precision.tolist() 
        recall = recall.tolist() 
        thresholds = thresholds.tolist()
    """
    return res

@router.post("/predict/SGDClassifierData")
def SGD_Classifier( filename: str = Form(...), percent: str = Form(...), loss: str = Form(...),
penalty: str = Form(...) ):
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float(percent)
    list = []

    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    y = column_or_1d(y, warn=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
    
    #判断有几个预测的目标
    for i in y:
        if i  == 0 or i == 1 or i == -1:
            if i not in list:
                list.append(i)
        else:
            auprc = 0
            break
    if len(list) != 2:
        auprc = 0
    if 0 in list and -1 in list:
        auprc = 0
    
    #AUPRC
    if auprc == 1:
        clf__ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
        y_score = clf__.decision_function( X )
        precision, recall, thresholds = precision_recall_curve( y, y_score )
        precision = precision.tolist() 
        recall = recall.tolist() 
        thresholds = thresholds.tolist()
    
    clf = SGDClassifier( loss=loss, penalty=penalty, max_iter=5 )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    per_test_data = clf.score( X_test, y_test )

    clf_ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    auroc = roc_auc_score(y, clf_.predict_proba(X)[:, 1], multi_class='ovo')
    
    per_test_data = format(per_test_data, '.4f')
    acc = format(acc, '.4f')
    auroc = format(auroc, '.4f')

    res["result_accuracy_of_test_data"] = per_test_data
    res['accuracy'] = acc
    res['auroc'] = auroc
    res["code"] = """
def SGDClassifierData( filename: str = Form(...), percent: str = Form(...), loss: str = Form(...),
penalty: str = Form(...) ):
    data = pd.read_csv('./static/data/'+ filename)
    per = float(percent)
    list = []

    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    y = column_or_1d(y, warn=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)
    
    #判断有几个预测的目标
    for i in y:
        if i  == 0 or i == 1 or i == -1:
            if i not in list:
                list.append(i)
        else:
            auprc = 0
            break
    if len(list) != 2:
        auprc = 0
    if 0 in list and -1 in list:
        auprc = 0
    
    #AUPRC
    if auprc == 1:
        clf__ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
        y_score = clf__.decision_function( X )
        precision, recall, thresholds = precision_recall_curve( y, y_score )
        precision = precision.tolist() 
        recall = recall.tolist() 
        thresholds = thresholds.tolist()
    
    clf = SGDClassifier( loss=loss, penalty=penalty, max_iter=5 )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    per_test_data = clf.score( X_test, y_test )

    clf_ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    auroc = roc_auc_score(y, clf_.predict_proba(X)[:, 1], multi_class='ovo')
    
    per_test_data = format(per_test_data, '.4f')
    acc = format(acc, '.4f')
    auroc = format(auroc, '.4f')
    """
    return res

@router.post("/predict/svc")
def SVC( filename: str = Form(...), percent: str = Form(...) ):
    #变量的初始化
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    per = float(percent)
    list = []
    auprc = 1
    auroc = ''
   
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    y = column_or_1d(y, warn=True).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)

    #判断有几个预测的目标
    for i in y:
        if i  == 0 or i == 1 or i == -1:
            if i not in list:
                list.append(i)
        else:
            auprc = 0
            break
    if len(list) != 2:
        auprc = 0
    if 0 in list and -1 in list:
        auprc = 0
    
    #AUPRC
    if auprc == 1:
        clf__ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
        y_score = clf__.decision_function( X )
        precision, recall, thresholds = precision_recall_curve( y, y_score )
        precision = precision.tolist() 
        recall = recall.tolist() 
        thresholds = thresholds.tolist()
    
    #ACC per_test_data
    clf = svm.SVC()
    clf.fit( X_train, y_train )
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    per_test_data = clf.score( X_test, y_test )
    
    #AUROC
    clf_ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    auroc = roc_auc_score(y, clf_.predict_proba(X)[:, 1], multi_class='ovo')

    per_test_data = format(per_test_data, '.4f')
    acc = format(acc, '.4f')
    auroc = format(auroc, '.4f')

    res["result_accuracy_of_test_data"] = per_test_data
    res["accuracy"] = acc
    res["auroc"] = auroc
    res["auprc"] = auprc
    res["precision"] = precision
    res["recall"] = recall
    res["thresholds"] = thresholds
    res["code"] = """ 
def SVC( filename: str = Form(...), percent: str = Form(...) ):
    #变量的初始化
    data = pd.read_csv('./static/data/'+ filename)
    per = float(percent)
    list = []
    auprc = 1
    auroc = ''

    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    y = column_or_1d(y, warn=True).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per, random_state=0)

    #判断有几个预测的目标
    for i in y:
        if i  == 0 or i == 1 or i == -1:
            if i not in list:
                list.append(i)
        else:
            auprc = 0
            break
    if len(list) != 2:
        auprc = 0
    if 0 in list and -1 in list:
        auprc = 0
    
    #AUPRC
    if auprc == 1:
        clf__ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
        y_score = clf__.decision_function( X )
        precision, recall, thresholds = precision_recall_curve( y, y_score )
        precision = precision.tolist() 
        recall = recall.tolist() 
        thresholds = thresholds.tolist()
    
    #ACC per_test_data
    clf = svm.SVC()
    clf.fit( X_train, y_train )
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    per_test_data = clf.score( X_test, y_test )
    
    #AUROC
    clf_ = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    auroc = roc_auc_score(y, clf_.predict_proba(X)[:, 1], multi_class='ovo')

    per_test_data = format(per_test_data, '.4f')
    acc = format(acc, '.4f')
    auroc = format(auroc, '.4f')
    """
    return res

#回归分析
@router.post("/predict/xgboost-regression")
def xgboost_regression( filename: str = Form(...), percent: str = Form(...) ):
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
    
    #per_train_data
    per_train_data = xlf.score(X_train, y_train)
    per_train_data = format(per_train_data, '.4f')

    #per_test_data
    per_test_data = xlf.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')

    y_pred = xlf.predict( X )

    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')
    res["mae"] = mae
    res["mse"] = mse
    res["r2"] = r2
    res["result_accuracy_of_test_data"] = per_test_data
    res["result_accuracy_of_train_data"] = per_train_data
    res["code"] = """
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
    
    per_train_data = xlf.score(X_train, y_train)
    per_train_data = format(per_train_data, '.4f')

    per_test_data = xlf.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')
    
    coef = xlf.coef_.tolist()
    intercept = xlf.intercept_.tolist()
    intercept[0] = format(intercept[0], '.4f') 
    while i < len(coef[0]):
        coef[0][i] = format(coef[0][i], '.4f')
        i += 1

    y_pred = xlf.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')
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

    per_train_data = reg.score(X_train, y_train)
    per_train_data = format(per_train_data, '.4f')
    
    per_test_data = reg.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')
    
    y_pred = reg.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')
    
    res["mae"] = mae
    res["mse"] = mse
    res["r2"] = r2

    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1):
        reg_list1[i] = float(reg_list1[i])
        reg_list1[i] = format(reg_list1[i], '.4f')
        i += 1
    res["result_coef"] = reg_list1
    res["result_intercept"] = reg_list2[0]
    res["result_accuracy_of_testdata"] = per_test_data
    res["result_accuracy_of_traindata"] = per_train_data
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
    per_train_data = format(per_train_data, '.4f')

    per_test_data = reg.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')

    y_pred = reg.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')

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

    per_train_data = reg.score(X_train, y_train)
    per_train_data = format(per_train_data, '.4f')

    per_test_data = reg.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')
    
    y_pred = reg.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')
    
    res["mae"] = mae
    res["mse"] = mse
    res["r2"] = r2

    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    res["result_accuracy_of_testdata"] = per_test_data
    res["result_accuracy_of_traindata"] = per_train_data
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

    y_pred = reg.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')
    
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
    per_train_data = format(per_train_data, '.4f')

    per_test_data = reg.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')

    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    
    y_pred = reg.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')
    
    res["mae"] = mae
    res["mse"] = mse
    res["r2"] = r2

    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    res["result_accuracy_of_testdata"] = per_test_data
    res["result_accuracy_of_traindata"] = per_train_data
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

    y_pred = reg.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')

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
    per_train_data = format(per_train_data, '.4f')

    per_test_data = reg.score(X_test, y_test)
    per_test_data = format(per_test_data, '.4f')
    
    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    
    y_pred = reg.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')
    
    res["mae"] = mae
    res["mse"] = mse
    res["r2"] = r2
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    res["result_accuracy_of_testdata"] = per_test_data
    res["result_accuracy_of_traindata"] = per_train_data
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
    
    y_pred = reg.predict( X )
    mae = mean_absolute_error(y, y_pred)
    mae = format(mae, '.4f')
    mse = mean_squared_error(y, y_pred)
    mse = format(mse, '.4f')
    r2 = r2_score(y, y_pred)
    r2 = format(r2, '.4f')
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


#####################################################
## New Version
#####################################################

@router.post("/regression/predict")
def train_regression_model( username: str = Form(...), percent: float = Form(...), method: str = Form(...)):
    # 读数据文件
    df = pd.read_csv(f'./static/data/{username}/data_zscore_fill_filter.csv')
    
    # 选择模型
    eval_metric = 'r2' # 默认使用r2作为评价指标
    if method == 'xgboost':
        model = xgb.XGBRegressor(verbosity=0, n_estimators=100, learning_rate=0.1)
        eval_metric = 'auc'
        # n_estimators – Number of gradient boosted trees. Equivalent to number of boosting rounds.
    
    # 划分训练集和测试集
    x = df.iloc[:, 1:]
    y = df.iloc[:, :1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent, random_state=42)
    
    # 训练模型，对于回归模型使用r2评价指标
    model.fit(x_train, y_train, eval_metric=eval_metric)
    
    # 在测试集上预测
    y_pred = model.predict(x_test)

    # format function returns a string
    mae = format(mean_absolute_error(y_test, y_pred), '.2f')
    mse = format(mean_squared_error(y_test, y_pred), '.2f')
    rmse = format(math.sqrt(float(mse)), '.2f')
    r2 = format(r2_score(y_test, y_pred), '.2f')

    res = [
        {'indicator': 'MAE', 'value': mae},
        {'indicator': 'MSE', 'value': mse},
        {'indicator': 'RMSE', 'value': rmse},
        {'indicator': 'R-squared', 'value': r2}
    ]

    # print(res)
    # print('aaaa',y_test.shape, type(y_test)) # (243,1) dataframe
    # print('bbb',y_pred.shape, type(y_pred)) # (243,) ndarray
    accuracy_res = calculate_regression_accuracy(np.squeeze(y_test.values), y_pred)
    res.extend(accuracy_res)

    return res


def calculate_regression_accuracy(y, y_pred): # gt & predicted value
    count_percent=[0,0,0,0] # 5%, 10%, 15%, 20%
    assert(len(y.shape)==len(y_pred.shape))
    data_length = y.shape[0] # y is the GT
    for i in range(data_length):
        accuracy = (y_pred[i] - y[i]) / y[i]
        for j in range(4):
            if accuracy <= (j+1)*0.05:
                count_percent[j] += 1
    count_percent = [(i / data_length) for i in count_percent]

    res = [
        {'indicator': '5%准确率', 'value': format(count_percent[0], '.2f')},
        {'indicator': '10%准确率', 'value': format(count_percent[1], '.2f')},
        {'indicator': '15%准确率', 'value': format(count_percent[2], '.2f')},
        {'indicator': '20%准确率', 'value': format(count_percent[3], '.2f')},
    ]

    return res


@router.post("/classification/predict")
def train_classification_model( username: str = Form(...), percent: float = Form(...), method: str = Form(...)):
    # 读数据文件
    df = pd.read_csv(f'./static/data/{username}/data_zscore_fill_filter.csv')

    # 划分训练集和测试集
    x = df.iloc[:, 1:]
    y = df.iloc[:, :1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent, random_state=42)

    # 选择模型并拟合
    if method == 'decision_tree':
        model = tree.DecisionTreeClassifier()
        # 训练模型，对于分类模型使用roc_auc评价指标
        model.fit(x_train, y_train)
    
    # 在测试集上预测
    y_pred = model.predict(x_test)

    # format function returns a string
    accuracy = format(accuracy_score(y_test, y_pred), '.2f')
    f1_score_result = format(f1_score(y_test, y_pred), '.2f')

    res = [
        {'indicator': '准确率', 'value': accuracy},
        {'indicator': 'F1 score', 'value': f1_score_result}
    ]
    return res