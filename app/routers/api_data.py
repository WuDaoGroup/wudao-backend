import os
import csv
import json
import base64

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd
import matplotlib.pyplot as plt
import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal
router = APIRouter(prefix="/data")
null = None
download_code = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                ]
            },
             {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                ]
            },
            {
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

@router.get("/{data_filename}_selected_feature.csv/features/info")
async def return_data_basic_file_info(data_filename: str):
    df = pd.read_csv(f"./static/data/{data_filename}_selected_feature.csv")
    len_df = len(df.index)
    content = []
    download_code['cells'][1]['source'] = []
    header = [
        {'key': 'name', 'value': 'name'},
        {'key': 'count', 'value': 'count'},
        {'key': 'missing_rate', 'value': 'missing_rate'},
        {'key': 'mean', 'value': 'mean'},
        {'key': 'max', 'value': 'max'},
        {'key': 'min', 'value': 'min'},
        {'key': 'std', 'value': 'std'},
        {'key': 'median', 'value': 'median'}
    ]
    for idx, e in enumerate(df.columns):
        h = {}
        h['name'] = e
        h['count'] = int(df[e].count())
        h['missing_rate'] = str(float((100-df[e].count()*100/len_df)))+"%"
        h['mean'] = float(df[e].mean())
        h['max'] = float(df[e].max())
        h['min'] = float(df[e].min())
        h['median'] = float(df[e].median())
        h['std'] = float(df[e].std())
        h['id'] = idx
        content.append(h)
    df_score = df.copy()
    df_score = (df_score-df_score.mean())/(df_score.std()+1e-12)
    df_score.to_csv(f'./static/data/{data_filename}_zscore.csv', index=False)
    code_content = """
df = pd.read_csv(f"./static/data/{data_filename}_selected_feature.csv")
len_df = len(df.index)
content = []
header = [
    {'key':'name','value':'name'},
    {'key':'count','value':'count'},
    {'key':'missing_rate','value':'missing_rate'},
    {'key':'mean','value':'mean'},
    {'key':'max','value':'max'},
    {'key':'min','value':'min'},
    {'key':'std','value':'std'},
    {'key':'median','value':'median'}
    ]
for idx, e in enumerate(df.columns):
    h = {}
    h['name'] = e
    h['count'] = int(df[e].count())
    h['missing_rate'] = str(float((100-df[e].count()*100/len_df)))+"%"
    h['mean'] = float(df[e].mean())
    h['max'] = float(df[e].max())
    h['min'] = float(df[e].min())
    h['median'] = float(df[e].median())
    h['std'] = float(df[e].std())
    h['id'] = idx
    content.append(h)
df_score = df.copy()
df_score = (df_score-df_score.mean())/(df_score.std()+1e-12)
df_score.to_csv(f'./static/data/{data_filename}_zscore.csv', index=False)

"""
    response = {
        'content': content,
        'header': header,
        'code': code_content
    }
    download_code['cells'][1]['source'].append(code_content)
    return response


@router.get("/{data_filename}_selected_feature.csv/zscore/type")
async def zscore_type(data_filename: str, selectType: str):
    df = pd.read_csv(f"./static/data/{data_filename}_zscore.csv")
    df_score = df.copy()
    code = """df = pd.read_csv(f"./static/data/{data_filename}_zscore.csv")
df_score = df.copy()"""
    if selectType == "均值填充":
        df_score.fillna(value=0, inplace=True)
        code = code + """df_score.fillna(value = 0,inplace=True)"""
    elif selectType == "中位数填充":
        h = []
        for idx, e in enumerate(df.columns):
            s = {}
            origin_mean = float(df[e].mean())
            origin_std = float(df[e].std())
            origin_median = float(df[e].median())
            fill_median = (origin_median - origin_mean)/(origin_std+1e-12)
            s['a'] = fill_median
            h.append(s)
        for idx, e in enumerate(df_score.columns):
            df_score[e].fillna(value=float(h[idx]['a']), inplace=True)
        code = code + """h = []
for idx, e in enumerate(df.columns):
    s={}
    origin_mean = float(df[e].mean())
    origin_std = float(df[e].std())
    origin_median = float(df[e].median())
    fill_median = (origin_median - origin_mean)/(origin_std+1e-12)
    s['a']= fill_median
    h.append(s)
for idx, e in enumerate(df_score.columns):
    df_score[e].fillna(value = float(h[idx]['a']),inplace=True)
    
    """
    else:
        raise HTTPException(status_code=240, detail="请选择")
    df_score.to_csv(
        f'./static/data/{data_filename}_zscore_fill.csv', index=False)
    res = df_score.to_json(orient="records")
    parsed = json.loads(res)
    code = code + """df_score.to_csv(f'./static/data/{data_filename}_zscore_fill.csv', index=False)
res = df_score.to_json(orient="records")
parsed = json.loads(res)
    """
    download_code['cells'][2]['source'].append(code)
    response = {
        'content': parsed,
        'code': code
    }
    return response


@router.get("/{data_filename}_selected_feature.csv/zscore/filter")
async def features_filter(data_filename: str, bar: float):
    df = pd.read_csv(f"./static/data/{data_filename}_zscore_fill.csv")
    for f in df.columns:
        df = df[abs(df[f]) < bar]
    df.to_csv(f'./static/data/{data_filename}_zscore.csv', index=False)
    df_score = pd.read_csv(f'./static/data/{data_filename}_zscore.csv')
    df_origin = pd.read_csv(
        f"./static/data/{data_filename}_selected_feature.csv")
    df_score = df_score*(df_origin.std()+1e-12)+df_origin.mean()
    df_score.to_csv(
        f'./static/data/{data_filename}_zscore_afterFilter.csv', index=False)
    # print (df_score)
    headers = []
    for idx, e in enumerate(df_score.columns):
        headers.append(e)
    print(headers)
    h_features = ['mean', 'max', 'min', 'median', 'std']
    num = 1
    for header in headers:
        plt.figure(figsize=(10, 6))
        plt.title(header, fontsize=18)
        plt.hist(df_score[header], bins=10, edgecolor='k', alpha=0.5)
        plt.savefig(
            f'./static/images/{data_filename}_selected_features_zscore_{num}.png')
        num += 1
    code= """df = pd.read_csv(f"./static/data/{data_filename}_zscore_fill.csv")
for f in df.columns:
    df = df[abs(df[f])<bar]
df.to_csv(f'./static/data/{data_filename}_zscore.csv', index=False)
df_score = pd.read_csv(f'./static/data/{data_filename}_zscore.csv')
df_origin = pd.read_csv(f"./static/data/{data_filename}_selected_feature.csv")
df_score = df_score*(df_origin.std()+1e-12)+df_origin.mean()
headers = []
for idx, e in enumerate(df_score.columns):
    headers.append(e)
print(headers)
h_features=['mean','max','min','median','std']
num = 1
for header in headers:
    plt.figure(figsize=(10,6))
    plt.title(header,fontsize=18)
    plt.hist(df_score[header],bins=10,edgecolor='k',alpha=0.5)
    plt.savefig(f'./static/images/{data_filename}_selected_features_zscore_{num}.png')
    num += 1"""
    response = {
        'content': df_score,
        'code': code
    }
    download_code['cells'][3]['source'].append(code)
    with open(f'./static/data/{data_filename}_download_code.ipynb', "w") as outfile:
        json.dump(download_code, outfile)
    return response
