import os,csv,json,base64

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal
from app.routers.api_file import selected_features
router = APIRouter(prefix = "/data")


@router.get("/{data_filename}_selected_feature.csv/features/info")
async def return_data_basic_file_info(data_filename: str):
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
        h['missing_rate'] = str(float((1-df[e].count()/len_df)*100))+"%"
        h['mean'] = float(df[e].mean())
        h['max'] = float(df[e].max())
        h['min'] = float(df[e].min())
        h['median'] = float(df[e].median())
        h['std'] = float(df[e].std())
        h['id'] = idx
        content.append(h)
    
    response={
        'content':content,
        'header':header
    }
    return response

@router.get("/{data_filename}_selected_feature.csv/zscore")
async def features_zscore(data_filename: str):
    df = pd.read_csv(f"./static/data/{data_filename}_selected_feature.csv")
    df_score = df.copy()
    df_score = (df_score-df_score.mean())/(df_score.std()+1e-12)
    df_score.to_csv(f'./static/data/{data_filename}_zscore.csv', index=False)
    res = df_score.to_json(orient="records")
    parsed = json.loads(res)
    response={
        'content': parsed,
    }
    return response


@router.get("/{data_filename}_selected_feature.csv/zscore/mean")
async def zscore_zero(data_filename: str):
    df = pd.read_csv(f"./static/data/{data_filename}_zscore.csv")
    df_score = df.copy()
    df_score.fillna(value = 0,inplace=True)
    df_score.to_csv(f'./static/data/{data_filename}_zscore_fill.csv', index=False)
    res = df_score.to_json(orient="records")
    parsed = json.loads(res)
    response={
        'content': parsed,
    }
    return response

@router.get("/{data_filename}_selected_feature.csv/zscore/median")
async def zscore_median(data_filename: str):
    df = pd.read_csv(f"./static/data/{data_filename}_selected_feature.csv")
    df_score = pd.read_csv(f"./static/data/{data_filename}_zscore.csv")
    h = []
    for idx, e in enumerate(df.columns):
        s={}
        origin_mean = float(df[e].mean())
        origin_std = float(df[e].std())
        origin_median = float(df[e].median())
        fill_median = (origin_median - origin_mean)/(origin_std+1e-12)
        s['a']= fill_median
        h.append(s)
    i = 0
    for idx, e in enumerate(df_score.columns):
        df_score[e].fillna(value = float(h[idx]['a']),inplace=True)
        print (idx)
    
    df_score.to_csv(f'./static/data/{data_filename}_zscore_fill.csv', index=False)
    res = df_score.to_json(orient="records")
    parsed = json.loads(res)
    response={
        'content': parsed,
    }
    return response


@router.get("/{data_filename}_selected_feature.csv/zscore/filter")
async def features_filter(data_filename: str, bar: float):
    df = pd.read_csv(f"./static/data/{data_filename}_zscore_fill.csv")
    for f in df.columns:
        # print("before ---------------------------------")
        # print()
        df = df[abs(df[f])<bar]
        print (f,df,"bb////////////////////////////")
    df.to_csv(f'./static/data/{data_filename}_zscore.csv', index=False)
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    response={
        'content': parsed,
    }
@router.get("/{data_filename}_selected_feature.png/features/info")
async def return_data_basic_image_info(data_filename: str):
    response = {
        'data' : len(selected_features)
    }
    # print(selected_features)
    return response