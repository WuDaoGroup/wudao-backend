import os

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal


router = APIRouter(prefix = "/models")

@router.post("/predict/lassoLars")
def lasso_lars( filename: str = Form(...), alpha: str = Form(...), normalize: str = Form(...)):
    i = 0
    normal = True
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    al = float(alpha)
    if normalize == 'True':
        normal = True
    else:
        normal = False
    reg = linear_model.LassoLars(alpha = al, normalize = normal)
    X = data.iloc[:, 1:]
    print(X)
    y = data.iloc[:, :1]
    print(y)
    reg.fit( X, y )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()
    print(reg_list1, reg_list2)
    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1):
        reg_list1[i] = float(reg_list1[i])
        reg_list1[i] = format(reg_list1[i], '.4f')
        i += 1
    res["result_coef"] = reg_list1
    res["result_intercept"] = reg_list2[0]
    return res

@router.post("/predict/ols")
def ordinary_least_squares(filename: str = Form(...)):
    i = 0
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    reg = linear_model.LinearRegression()
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    reg.fit( X, y )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()
    print(reg_list1, reg_list2)
    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    return res

@router.post("/predict/lasso")
def lasso( filename: str = Form(...), alpha: str = Form(...)):
    i = 0
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    alpha = float(alpha)
    reg = linear_model.Ridge(alpha)
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    reg.fit( X, y )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()
    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    return res

@router.post("/predict/ridge_regression")
def ridge_regression( filename: str = Form(...), alpha: str = Form(...)):
    i = 0
    res = {}
    data = pd.read_csv('./static/data/'+ filename)
    alpha = float(alpha)
    reg = linear_model.Ridge(alpha)
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    reg.fit( X, y )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()
    reg_list2[0] = format(reg_list2[0], '.4f') 
    while i < len(reg_list1[0]):
        reg_list1[0][i] = format(reg_list1[0][i], '.4f')
        i += 1
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
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
    return res
