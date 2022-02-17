import os

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal

router = APIRouter(prefix = "/files")
@router.post("/upload")
async def upload_file(upload_file: UploadFile = File(...)):
    file_type=os.path.splitext(upload_file.filename)[1]
    if file_type not in [".csv", ".xls", ".xlsx"]:
        raise HTTPException(status_code=240, detail="File type not correct")
    file_location = f"./static/data/{upload_file.filename}"
    print(file_location)
    with open(file_location, "wb+") as file_object:
        file_object.write(upload_file.file.read())
    return {"info": f"file '{upload_file.filename}' saved at '{file_location}'"}
