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
@router.post("/predict/ols")
def ordinary_least_squares(filename: str = Form(...)):
    res = {}
    data = pd.read_csv('./data/'+ filename)
    reg = linear_model.LinearRegression()
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    reg.fit( X, y )
    reg_list1 = reg.coef_.tolist()
    reg_list2 = reg.intercept_.tolist()
    res["result_coef"] = reg_list1[0]
    res["result_intercept"] = reg_list2[0]
    return res

@router.post("/predict/bdtr")
def boosted_decision_tree_regression( filename: str = Form(...), id: str = Form(...)):
    the_addr = 'C:\\Users\\DELL\\Desktop\\wudao-backend\\pictures\\' + filename + id + '_handle_result.png'
    res = {}
    data = pd.read_csv('./data/'+ filename) 
    rng = np.random.RandomState(1)
    X = data.iloc[:, 1:2]
    y = data.iloc[:, 0:1]
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
    plt.savefig(the_addr)
    
    res['pic_addr'] = the_addr
    return res
