import os,csv,json
import pathlib

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal

# -*- coding: UTF-8 -*-

router = APIRouter(prefix = "/explanation")

# 降维
@router.post("/reduction")
async def data_dimension_reduction(username: str = Form(...), method: str = Form(...), dimension: int = Form(...), target: str = Form(...)):
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    df = pd.read_csv(f'./static/data/{username}/data_zscore_fill_filter.csv')
    if method=='PCA': # 判断降维类别
        reduction_model = PCA().fit_transform(df)
    elif method=='TSNE':
        reduction_model = TSNE(n_components=dimension, learning_rate='auto').fit_transform(df)
    df_target=df[target]
    if dimension ==2: # 判断降维维度
        print(df_target.shape, reduction_model.shape)
        df_subset = pd.DataFrame({'2d-one': reduction_model[:,0], '2d-two': reduction_model[:,1], 'target': df_target})
        plt.figure(figsize=(6,6))
        sns.scatterplot(
            x="2d-one", y="2d-two",
            hue="target",
            # palette=sns.color_palette("hls", df_target.shape[0]),
            palette=sns.color_palette('coolwarm', as_cmap = True), 
            data=df_subset,
            legend=False,
            alpha=0.3,
        )
    elif dimension==3:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        df_subset = pd.DataFrame({'3d-one': reduction_model[:,0], '3d-two': reduction_model[:,1], '3d-three': reduction_model[:,2], 'target': df_target})
        ax.scatter(
            xs=df_subset["3d-one"], 
            ys=df_subset["3d-two"], 
            zs=df_subset["3d-three"], 
            c=df_subset["target"], 
            cmap='tab10'
        )
        ax.set_xlabel('3d-one')
        ax.set_ylabel('3d-two')
        ax.set_zlabel('3d-three')
    plt.title(f'Reduction Result of {method} with {dimension} dimensions', fontsize=18)
    pathlib.Path(f'./static/data/{username}/images/explanation').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'./static/data/{username}/images/explanation/reduction_{method}_{dimension}_{target}.png')
    res = {'message': 'reduction success'}
    return res

# 特征的相关矩阵
@router.post("/correlation/feature")
async def feature_correlation(username: str = Form(...), method: str = Form(...)): 
    df = pd.read_csv(f'./static/data/{username}/data_zscore_fill_filter.csv')
    corr_mat = df.corr(method = method)
    plt.subplots(figsize=(24, 16))
    sns.heatmap(corr_mat, square=True)
    plt.title(f'Correlation Matrix of Features ({method})', fontsize=18)
    pathlib.Path(f'./static/data/{username}/images/explanation').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'./static/data/{username}/images/explanation/correlation_feature_{method}.png')
    response = {'message': 'generate feature correlation success'}
    return response

@router.post("/correlation/target")
async def target_correlation(username: str = Form(...), k_number: int = Form(...), target: str = Form(...)):
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # sns.set(font='SimHei')  # 解决Seaborn中文显示问题 

    df = pd.read_csv(f'./static/data/{username}/data_zscore_fill_filter.csv')
    # 默认使用spearman相关性
    corr_mat = df.corr(method = 'spearman')
    # nlargest可以用于找到列表中最大的前k_number个元素
    cols = corr_mat.nlargest(k_number, target)[target].index
    cm = np.corrcoef(df[cols].values.T)
    plt.subplots(figsize=(24, 16))
    sns.heatmap(cm, annot=True, square=False, yticklabels=cols.values, xticklabels=cols.values)
    plt.title(f'Correlation Matrix of Targets (Top {k_number} related features)', fontsize=18)
    pathlib.Path(f'./static/data/{username}/images/explanation').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'./static/data/{username}/images/explanation/correlation_target_{target}_{k_number}.png')
    response = {'message': 'generate target correlation success'}
    return response

@router.post("/correlation/pairwise")
async def pairwise_feature_correlation(username: str = Form(...)):
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # sns.set(font='SimHei')  # 解决Seaborn中文显示问题
    # features = features.split(',')
    df = pd.read_csv(f'./static/data/{username}/data_zscore_fill_filter.csv')
    features = df.columns
    sns.pairplot(df[features], height = 2.5)
    pathlib.Path(f'./static/data/{username}/images/explanation').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'./static/data/{username}/images/explanation/correlation_feature_pairwise.png')
    response = {'message': 'generate pairwise feature correlation success'}
    return response