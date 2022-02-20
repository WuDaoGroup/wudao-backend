import os,csv,json

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal


router = APIRouter(prefix = "/data")


@router.get("/{data_filename}_selected_feature.csv/features/info")
async def return_data_basic_file_info(data_filename: str):
    df = pd.read_csv(f"./static/data/{data_filename}_selected_feature.csv")
    len_df = len(df.index)
    content = []
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
    }
    return response