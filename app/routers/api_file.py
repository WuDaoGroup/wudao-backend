import os

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd

import app.schemas as schemas
import app.crud as crud
from app.database import SessionLocal

router = APIRouter(prefix = "/files")
@router.get("/upload")
async def upload_file(upload_file: UploadFile = File(...)):
    file_type=os.path.splitext(upload_file.filename)[1]
    if file_type not in [".csv", ".xls", ".xlsx"]:
        raise HTTPException(status_code=240, detail="File type not correct")
    file_location = f"./data/{upload_file.filename}"
    print('get_file:',file_location)
    with open(file_location, "wb+") as file_object:
        file_object.write(upload_file.file.read())
    data = pd.read_csv(f"./data/{upload_file.filename}",engine = "python",header=None)
    df = pd.DataFrame(data)
    print(df)
    return {"info": f"file '{upload_file.filename}' saved at '{file_location}'"}

@router.get("/download/{file_addr}")
def get_image(file_addr: str):
    return FileResponse(path=file_addr, filename=file_addr)