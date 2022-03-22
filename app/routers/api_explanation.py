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

@router.post("/reduction")
async def data_dimension_reduction(username: str = Form(...), method: str = Form(...), dimension: int = Form(...), target: str = Form(...)):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
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

@router.post("/{data_filename}/feature_corr")
async def return_feature_corr(data_filename: str, methods: str):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei')  # 解决Seaborn中文显示问题
    
    df = pd.read_csv(f"./static/data/{data_filename}_zscore_fill_filter.csv")
    corr_mat = df.corr(method = methods)
    f, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(corr_mat, vmax=.8, square=True, ax=ax)

    plt.savefig(f'./static/images/{data_filename}_feature_corr_img.png') # 存储图片
    response={
        'pic_addr': './static/images/{data_filename}_feature_corr_img.png'
    }
    return response

@router.post("/{data_filename}/object_matrix")
async def return_object_matrix(info:schemas.FeatureCorrFeaturesInfo, data_filename: str,):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei')  # 解决Seaborn中文显示问题 

    df = pd.read_csv(f"./static/data/{data_filename}_zscore_fill_filter.csv")

    corr_mat = df.corr(method = 'spearman')

    k = int(info.k_number)
    cols = corr_mat.nlargest(k, info.object)[info.object].index
    #nlargest可以用于找到列表中最大的前k个元素
    cm = np.corrcoef(df[cols].values.T)
    plt.figure(figsize=(15, 8))
    sns.heatmap(cm, annot=True, square=True, yticklabels=cols.values, xticklabels=cols.values)

    plt.savefig(f'./static/images/{data_filename}_object_matrix_img.png') # 存储图片
    response={
        'pic_addr': './static/images/{data_filename}_object_matrix_img.png'
    }
    return response

@router.post("/{data_filename}/pairwise_feature_corr")
async def return_pairwise_feature_corr(cols: list, data_filename: str):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei')  # 解决Seaborn中文显示问题 
    df = pd.read_csv(f"./static/data/{data_filename}_zscore_fill_filter.csv")
    sns.pairplot(df[cols], height = 2.5)
    plt.savefig(f'./static/images/{data_filename}_pairwise_feature_corr_img.png') # 存储图片
    response={
        'pic_addr': './static/images/{data_filename}_pairwise_feature_corr_img.png'
    }
    return response