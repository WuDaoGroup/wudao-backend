import os,csv,json

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

@router.post("/{data_filename}/dimension_reduction")
async def return_dimension_reduction(info: schemas.ExplanationInfo,data_filename: str):
    df = pd.read_csv(f"./static/data/{data_filename}_zscore_afterFilter.csv")
    print(df)
    data_array = np.array(df)
    if info.type=='PCA': # 判断降维类别
        reduction_model = PCA().fit_transform(data_array)
    elif info.type=='TSNE':
        reduction_model = TSNE(n_components=int(info.dimmension), learning_rate=int(info.learningrate)).fit_transform(data_array)
    target=df[info.target].tolist()
    if info.dimmension=='2': # 判断降维维度
        plt.figure(figsize=(15, 8))
        plt.subplot(121)
        plt.scatter(reduction_model[:, 0], reduction_model[:, 1], c = target)
    elif info.dimmension=='3':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduction_model[:, 0], reduction_model[:, 1],reduction_model[:, 2], c=target)
    plt.savefig(f'./static/images/{data_filename}_dimension_reduction_img.png') # 存储图片

    response={
        'pic_addr': './static/images/{data_filename}_dimension_reduction_img.png'
    }
    return response

@router.post("/{data_filename}/feature_corr")
async def return_feature_corr(data_filename: str, methods: str):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei')  # 解决Seaborn中文显示问题
    
    df = pd.read_csv(f"./static/data/{data_filename}_zscore_afterFilter.csv")
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

    df = pd.read_csv(f"./static/data/{data_filename}_zscore_afterFilter.csv")

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
    df = pd.read_csv(f"./static/data/{data_filename}_zscore_afterFilter.csv")
    sns.pairplot(df[cols], height = 2.5)
    plt.savefig(f'./static/images/{data_filename}_pairwise_feature_corr_img.png') # 存储图片
    response={
        'pic_addr': './static/images/{data_filename}_pairwise_feature_corr_img.png'
    }
    return response