import os,csv,json

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal


router = APIRouter(prefix = "/files")
@router.post("/upload")
async def create_upload_file(upload_file: UploadFile = File(...)):
    file_type=os.path.splitext(upload_file.filename)[1]
    if file_type not in [".csv", ".xls", ".xlsx"]:
        raise HTTPException(status_code=240, detail="File type not correct")
    file_location = f"./static/data/{upload_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(upload_file.file.read())


    
@router.get("/{data_filename}/analysis/content")
async def return_data_file_info(data_filename: str):
    if data_filename.endswith(".csv"):
        df = pd.read_csv(f"./static/data/{data_filename}")
    else:
        df = pd.read_excel(f"./static/data/{data_filename}")
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    for idx, p in enumerate(parsed):
        p['id'] = idx
    # print(parsed)
    header = []
    for idx, e in enumerate(df.columns):
        h = {}
        h['key'] = e
        h['value'] = e
        header.append(h)
        
    response={
        'header': header,
        'content': parsed,
    }
    return response

@router.post("/{data_filename}/features/info")
async def process_selected_features(info: list[schemas.FeatureInfo], data_filename: str):
    if data_filename.endswith(".csv"):
        df = pd.read_csv(f"./static/data/{data_filename}")
    else:
        df = pd.read_excel(f"./static/data/{data_filename}")
    selected_features=[]
    for i in info:
        if i.type=="target":
            selected_features.append(i.value)
    for i in info:
        if i.type=="feature":
            selected_features.append(i.value)
    df = df[selected_features]
    df.to_csv(f'./static/data/{data_filename}_selected_feature.csv', index=False)
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    response={
        'header': selected_features,
        'content': parsed,
    }
    return response